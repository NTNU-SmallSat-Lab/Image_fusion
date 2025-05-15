import hypso
import hypso.write
from utilities import Get_path
import cv2
import numpy as np
import os
import json

# Initialize a Path object
file_path, name = Get_path("L1A file")
#file_path = "data\\bluenile_2025-01-25T08-23-16Z-l1a.nc"

sat_obj = hypso.Hypso2(file_path, verbose=True)
print(sat_obj.l1a_cube.shape)
sat_obj.generate_l1b_cube() #Note that filepaths on lines 905 and 913 are changed to hypso2 files
#sat_obj.generate_geometry()
hypso.write.l1b_nc_writer.write_l1b_nc_file(sat_obj)

#Nominal: 956, Full: 1092