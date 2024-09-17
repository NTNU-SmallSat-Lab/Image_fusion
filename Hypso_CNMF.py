import numpy as np
import saver as sv
import utilities as util
import os
from CNMF import CNMF, Get_VCA
import loader as ld
import matplotlib.pyplot as plt
from PIL import Image


data_string, name = util.Get_path()

rgb = {
    "R": 72,
    "G": 43,
    "B": 15
}

endmember_count = 2
"""x_start = int(input("x_start: "))
x_end = int(input("x_end: "))
y_start = int(input("y_start"))
y_end = int(input("y_end"))"""

x_start, x_end, y_start, y_end = 0, 400, 0, 400
pix_coords = [x_start,x_end,y_start,y_end]

VCA_init = Get_VCA(data_string, endmember_count)

full_arr = ld.load_l1b_cube(data_string)

arr = full_arr[x_start:x_end,y_start:y_end,:]
arr = arr/arr.max()
bands = full_arr.shape[2]
size = (x_end-x_start,y_end-y_start)
downsample_factor = 4
sigma = 1

rgb_representation = arr[:,:,[rgb["R"],rgb["G"],rgb["B"]]] #Generate RGB representation of original MSI

lowres_downsampled = util.Downsample(arr, sigma=sigma, downsampling_factor=downsample_factor) #Generate downsampled HSI
upsized_image = np.repeat(np.repeat(lowres_downsampled, downsample_factor, axis=0), downsample_factor, axis=1)

spatial_transform_matrix = util.Gen_downsampled_spatial(downsample_factor,size).transpose() #Generate spatial transform for downsampling

spectral_response_matrix = util.Gen_spectral(rgb=rgb, bands=bands, spectral_spread=3)

Upscaled_datacube, endmembers, abundances = CNMF(lowres_downsampled, rgb_representation, spatial_transform_matrix, spectral_response_matrix, VCA_init, endmember_count)
Upscaled_datacube = Upscaled_datacube/Upscaled_datacube.max()
arr = arr/arr.max()
error = util.get_error(Upscaled_datacube,arr)
error_rgb = np.stack([error] * 3, axis=-1)

top = np.hstack([arr[:,:,[72, 43, 15]],upsized_image[:,:,[72, 43, 15]]])
bottom = np.hstack([Upscaled_datacube[:,:,[72, 43, 15]],error_rgb])
final_image = np.vstack([top,bottom])

spec_error = util.get_spectral_error(Upscaled_datacube, arr)

save_path = f"outputs\\{name}_{x_start}-{x_end}x_{y_start}-{y_end}y_{endmember_count}EM\\"

if not os.path.exists(save_path):
        os.mkdir(save_path)

Image.fromarray((final_image*255).astype(np.uint8)).save(f"{save_path}\\final_image.png")

sv.save_spec_error(spec_error=spec_error, path=save_path)
sv.save_endmembers(endmembers, abundances, size, save_path)