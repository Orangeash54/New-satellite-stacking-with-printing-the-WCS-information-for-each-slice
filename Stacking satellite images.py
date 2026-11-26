#%%
import os
import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from reproject import reproject_interp
from skyfield.api import EarthSatellite, load, wgs84

warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", message=".*All-NaN slice.*")

folder_path = Path(r"C:\Users\orang\OneDrive - University of Warwick\Desktop\29486 - Copy")
max_pointing_drift_arcmin = 5.0
max_time_gap_sec = 300.0

# Basically groups each satellite observation by recording when the telescope is just looking at 1 direction at one time

# Defining the small patch around the satellite
CUTOUT_SIZE = 301              # odd so that their is 1 central pixel, That hopefully the satellite is at
INTERP_ORDER = "bilinear"

# Timing factors that we use to define the satellite position + a possible extra constant to possibly match data better (maybe account for light travel time)
READOUT_BOTTOM_FIRST = True    # Read from the bottom of the image up
TIME_OFFSET_SEC = -0.25          # possible extra constant
N_TIME_ITER = 3                # Number of iterations we use to define the position of the satellite
ROW_PERIOD_SEC = 39.027e-6 * 3
# Getting the data from Space-Track.org for 13th November
tle_file = Path(r"C:\Users\orang\OneDrive - University of Warwick\Desktop\Data for 13th November GPS satellites.json")
NORAD_ID = 29486

ts = load.timescale()

# All orbit datasets for our satellite (Space-Track publishes about one per day)
with open(tle_file, "r", encoding="utf-8") as f:
    tle_sets = [EarthSatellite(r["TLE_LINE1"], r["TLE_LINE2"], r["OBJECT_NAME"], ts)
                for r in json.load(f) if int(r["NORAD_CAT_ID"]) == NORAD_ID]
print(f"Loaded {len(tle_sets)} orbit datasets for {tle_sets[0].name}")

def closest_tle(astropy_time):
    # orbit dataset whose epoch is closest to the given time
    t_sf = ts.from_astropy(astropy_time)
    return min(tle_sets, key=lambda s: abs(s.epoch - t_sf))

# our position
bluffdon = wgs84.latlon(52.376868, -1.567202, elevation_m=94.0)

# function to get the satellites RA and DEC
def get_sat_radec(sat, observer, astropy_time):
    t_sf = ts.from_astropy(astropy_time)
    topocentric = (sat - observer).at(t_sf)
    ra, dec, _ = topocentric.radec()
    return ra._degrees, dec.degrees


def satellite_position_in_frame(sat, observer, wcs, t_start, exptime, row_period_sec, h):
    # first guess: middle of the frame
    t = t_start + (TIME_OFFSET_SEC + 0.5 * exptime + 0.5 * h * row_period_sec) * u.s
    for _ in range(N_TIME_ITER):
        ra, dec = get_sat_radec(sat, observer, t)
        _, y = wcs.world_to_pixel_values(ra, dec)
        row = float(np.clip(y, 0, h - 1))
        readout_row = row if READOUT_BOTTOM_FIRST else (h - 1) - row
        t = t_start + (TIME_OFFSET_SEC + 0.5 * exptime + readout_row * row_period_sec) * u.s
    ra, dec = get_sat_radec(sat, observer, t)
    x, y = wcs.world_to_pixel_values(ra, dec)
    return t, float(ra), float(dec), float(x), float(y)
# This is the main function which calculates the satellites RA and DEC and its pixel position in x and y on the origional images as well as the exact time.
# It does this by taking the time of the image, then adding an arbitrary time offset that we can define later if needed to fit data
# better, 0.5* the exposure time for, then adding to h (the height of the row)* by the row period which is 117 micro seconds per row.
# It iterates through this procedure to find the time and x y pixel coordinate 3 times to give the most accurate value

def make_satellite_grid(sat_ra_deg, sat_dec_deg, ref_wcs, size):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [sat_ra_deg, sat_dec_deg]
    w.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]      # FITS 1-based -> 0-based index (size-1)/2
    w.wcs.cd = ref_wcs.celestial.pixel_scale_matrix     # deg/pixel, includes rotation
    w.pixel_shape = (size, size)
    return w
# this function sets the WCS of each of the images and zooms in on an area of 201x201 pixel grid where each pixels RA and DEC is defined by the WCS

# SCAN FILES & GROUP INTO PASSES
all_files = [str(p) for p in folder_path.glob("*.fit*") if p.is_file() and not p.name.startswith("stacked_")]


metadata_list = []
for fp in all_files:
    with fits.open(fp) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        hdr = hdu.header
        h, w = hdu.data.shape
        wcs = WCS(hdr)
        t_obs = Time(hdr["DATE-OBS"], format="isot", scale="utc")
        ra_c, dec_c = wcs.pixel_to_world_values(w / 2.0, h / 2.0)
        coord = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
        metadata_list.append({"path": fp, "time": t_obs, "coord": coord})

metadata_list.sort(key=lambda x: x["time"])

passes = [[metadata_list[0]]]
for item in metadata_list[1:]:
    prev = passes[-1][-1]
    same_pointing = prev["coord"].separation(item["coord"]).arcminute <= max_pointing_drift_arcmin
    close_in_time = (item["time"] - prev["time"]).sec <= max_time_gap_sec
    if same_pointing and close_in_time:
        passes[-1].append(item)
    else:
        passes.append([item])

print(f"Found {len(passes)} distinct telescope passes.\n")

# basically just groups the data into seperate passes by recording when the telescope is just looking in 1 direction

# IDENTIFY & REPROJECT-STACK EACH PASS
stacks = []

for pass_idx, pass_items in enumerate(passes, 1):
    print(f"=== PASS {pass_idx}/{len(passes)} ({len(pass_items)} frames) ===")
    if len(pass_items) < 2:
        continue

# Reference WCS (pixel scale + rotation) from the middle frame of the pass
    mid_path = pass_items[len(pass_items) // 2]["path"]
    with fits.open(mid_path) as hdul:
        ref_wcs = WCS(next(h for h in hdul if h.data is not None).header)

    satellite = closest_tle(pass_items[0]["time"])
    print("CD matrix:", ref_wcs.celestial.pixel_scale_matrix.tolist())
    cutouts = []

    frame_rows = []          # per-frame record: time, predicted RA/Dec, predicted pixel in original frame

    for item in pass_items:
        with fits.open(item["path"]) as hdul:
            hdu = next(h for h in hdul if h.data is not None)
            data = hdu.data.astype(float)
            hdr = hdu.header
        h, w = data.shape
        wcs_in = WCS(hdr).celestial

        t_start = Time(hdr["DATE-OBS"], format="isot", scale="utc")
        exptime = float(hdr.get("EXPTIME", 0.5))

        # reading what is needed from earlier functions

        t_sat, ra_sat, dec_sat, px_x, px_y = satellite_position_in_frame(
            satellite, bluffdon, wcs_in, t_start, exptime, ROW_PERIOD_SEC, h)

        # Background-subtract the stars from each image (so they dont clutter visual)
        data_sub = data - np.nanmedian(data[10:2120, 20:3190])

        # Satellite-centred grid for THIS frame, then resample onto it
        wcs_out = make_satellite_grid(ra_sat, dec_sat, ref_wcs, CUTOUT_SIZE)
        cutout, footprint = reproject_interp((data_sub, wcs_in), wcs_out,
                                             shape_out=(CUTOUT_SIZE, CUTOUT_SIZE),
                                             order=INTERP_ORDER)
        cutout[footprint == 0] = np.nan
        cutouts.append(cutout)
        print(f"  slice {len(cutouts):3d}: start {t_start.isot}  centre RA {ra_sat:.6f}  Dec {dec_sat:.6f}")
        frame_rows.append((t_sat.isot, ra_sat, dec_sat))
    #gets rid of pixels with no data that may arise from the repoject plotting

    cube = np.array(cutouts)
    stacked_image = np.nanmedian(cube, axis=0)

    # this is what stacks the image, it stores each cutout image in a matrix containing the frame, row and column data
    # then it takes for each pixel position the median of its values for each pass resulting in 1 201x201 stacked image of the satellite

    # Reference time = middle usable frame. Its predicted position defines the
    # WCS of the stacked image, so pixel_to_world on the stack gives
    # predicted position at T_REF + mean offset of the satellite from prediction.
    ref_row = frame_rows[len(frame_rows) // 2]
    t_ref, ra_ref, dec_ref = ref_row[0], ref_row[1], ref_row[2]
    wcs_stack = make_satellite_grid(ra_ref, dec_ref, ref_wcs, CUTOUT_SIZE)

    # basically every cutout is now centred on a point that our iteration formula predicts where the satellite will be at that time, then the images are stacked on top of each other

    out_hdr = wcs_stack.to_header()
    out_hdr["T_REF"] = (t_ref, "UTC time the WCS centre refers to")
    out_hdr["SATNAME"] = satellite.name
    out_hdr["NORAD"] = satellite.model.satnum
    out_hdr["NFRAMES"] = (len(cutouts), "frames in median stack")
    out_hdr["TOFFSET"] = (TIME_OFFSET_SEC, "extra time offset applied [s]")
    out_hdr["RDBOTTOM"] = (READOUT_BOTTOM_FIRST, "row 0 read out first")

    timestamp_tag = pass_items[0]["time"].strftime("%Y%m%d_%H%M%S")
    out_file = Path(r"C:\Users\orang\OneDrive - University of Warwick\Desktop\Satellite Images") / f"stacked_{NORAD_ID}_pass{pass_idx}_{timestamp_tag}.fits"
    fits.writeto(out_file, stacked_image.astype(np.float32), out_hdr, overwrite=True)
    cube_file = out_file.with_name(out_file.name.replace("stacked_", "cube_"))
    fits.writeto(cube_file, cube.astype(np.float32), overwrite=True)
    stacks.append((out_file.name, stacked_image))
    print(f"Saved -> {out_file.name}  ({len(cutouts)} frames, T_REF {t_ref})\n")

# PLOT: zoomed-in view of the centre of each stack, crosshair = predicted position
zoom = 100
c = CUTOUT_SIZE // 2
fig, axes = plt.subplots(1, len(stacks), figsize=(5 * len(stacks), 5), squeeze=False)
for ax, (name, img) in zip(axes[0], stacks):
    crop = img[c - zoom:c + zoom + 1, c - zoom:c + zoom + 1]
    med, std = np.nanmedian(crop), np.nanstd(crop)
    ax.imshow(crop, cmap="gray", origin="lower", vmin=med - 2 * std, vmax=med + 8 * std)
    ax.axhline(zoom, color="cyan", ls="--", lw=0.8)
    ax.axvline(zoom, color="cyan", ls="--", lw=0.8)
    ax.set_title(name, fontsize=9)
plt.tight_layout()
plt.show()
