import hypso
import hypso.calibration
from utilities import Get_path
import cv2
import numpy as np
import os
import json

# Initialize a Path object
file_path, name = Get_path("L1A file")
#file_path = "data\\bluenile_2025-01-25T08-23-16Z-l1a.nc"
meta_path = str(file_path).replace("l1a.nc", "meta.json")

with open(meta_path, 'r') as file:
    metadata = json.load(file)

# Extract the values (adjust keys if nested differently)
gsd_along = metadata.get('gsd_along')
gsd_across = metadata.get('gsd_across')

sat_obj = hypso.Hypso1(file_path, verbose=True)
sat_obj.generate_l1b_cube() #Note that filepaths on lines 905 and 913 are changed to hypso2 files
#sat_obj.generate_geometry()
sat_obj.write_l1b_nc_file(overwrite=True)

#Nominal: 956, Full: 1092

R, G, B = np.sum(sat_obj.l1b_cube[:,:,58:61],axis=2), np.sum(sat_obj.l1b_cube[:,:,69:72],axis=2), np.sum(sat_obj.l1b_cube[:,:,87:91],axis=2)
rgb = np.stack([R,G,B], axis=-1)

normalized = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX)
normalized = np.uint8(normalized)  # Convert to uint8 for saving

bgr_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)

dx = gsd_along/gsd_across
dy = 1.0
height, width = bgr_image.shape[:2]

if dx < dy:
    scale_factor = dy / dx
    new_width = int(width * scale_factor)
    new_height = height
else:
    scale_factor = dx / dy
    new_width = width
    new_height = int(height * scale_factor)
    
corrected_image = cv2.resize(bgr_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

rgb = np.asarray(corrected_image)[:,:,::-1] #Scaled datacube

cv2.imwrite("slice.png", corrected_image)  # Save as JPG or PNG