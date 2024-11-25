from hypso.hypso1 import Hypso1
from utilities import Get_path

# Initialize a Path object
file_path, name = Get_path()

HSI_file = Hypso1(file_path)
HSI_file._run_calibration()
HSI_file._run_geometry()
HSI_file.write_l1b_nc_file()