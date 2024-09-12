from hypso import Hypso
import numpy as np
from saver import save_RGB
from utilities import Wavelength_to_band, Get_path, flatten_datacube
import os

# Initialize a Path object
file_path, name = Get_path()

HSI_file = Hypso(file_path)
raw_data = HSI_file.l1b_cube.astype(np.float64)
mini = True
if mini:
        raw_data = raw_data[300:500,300:500,:]
        name = f"{name}_mini"
raw_data = raw_data.reshape(raw_data.shape[0]*raw_data.shape[1],raw_data.shape[2])

if not os.path.exists(".\\hypso_1_datacubes\\"):
        os.mkdir(".\\hypso_1_datacubes\\")
filename = f"hypso_1_datacubes\\{name}.txt"
np.savetxt(X=raw_data,fname=filename)