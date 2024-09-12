from hypso import Hypso
import numpy as np
from saver import save_RGB
from utilities import Wavelength_to_band, Get_path, flatten_datacube
import os

# Initialize a Path object
file_path, name = Get_path()

HSI_file = Hypso(file_path)
raw_data = HSI_file.l1b_cube.astype(np.float64)
raw_data = raw_data.reshape(raw_data.shape[0]*raw_data.shape[1],raw_data.shape[2])
print(raw_data.shape)
if not os.path.exists(".\\hypso_1_datacubes\\"):
        os.mkdir(".\\hypso_1_datacubes\\")
filename = f"hypso_1_datacubes\\{name}.txt"
np.savetxt(X=raw_data,fname=filename)

#RGB_representation = raw_data[:,:,[Wavelength_to_band(650, HSI_file), Wavelength_to_band(550, HSI_file), Wavelength_to_band(450, HSI_file)]] #bands R:75, G:46, B:18

data = HSI_file.l1b_cube