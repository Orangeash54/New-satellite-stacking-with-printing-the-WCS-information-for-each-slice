from astropy.wcs import WCS

CD = [[0.000170105674503, 1.31845652278e-06], [-1.29560268552e-06, 0.000170137469177]]   # CD matrix
centre_ra = 287.433582                                  # centre RA
centre_dec = 31.304375                                # Dec

# read these off DS9
pixel_x = 187
pixel_y = 116
#inputs for x and y pixels
CUTOUT_SIZE = 301

w = WCS(naxis=2)
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
w.wcs.crval = [centre_ra, centre_dec]
w.wcs.crpix = [(CUTOUT_SIZE + 1) / 2, (CUTOUT_SIZE + 1) / 2]
w.wcs.cd = CD

ra, dec = w.pixel_to_world_values(pixel_x - 1, pixel_y - 1)
print(f"RA = {ra:.6f} deg   Dec = {dec:.6f} deg")