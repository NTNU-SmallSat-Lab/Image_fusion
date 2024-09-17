import utilities as util
import numpy as np
import netCDF4 as nc
import loader as ld

data_string, name = util.Get_path()

arr = ld.load_l1b_cube(data_string)