import numpy as np
from Plotting import save_final_image, save_endmembers, Normalize
import utilities as util
import os
from CNMF import CNMF, Get_VCA
import loader as ld
from Test import range


data_string, name = util.Get_path()

rgb = [72, 43, 15]

endmember_count = 20
"""x_start = int(input("x_start: "))
x_end = int(input("x_end: "))
y_start = int(input("y_start"))
y_end = int(input("y_end"))"""

x_start, x_end, y_start, y_end = 0, 200, 0, 200
pix_coords = [x_start,x_end,y_start,y_end] #only needed if taking VCA of section, not entire image

VCA_init = Get_VCA(data_string, endmember_count)

full_arr = ld.load_l1b_cube(data_string)

arr = full_arr[x_start:x_end,y_start:y_end,:]

size = (x_end-x_start,y_end-y_start)
downsample_factor = 4
sigma = 1

lowres_downsampled = util.Downsample(arr, sigma=sigma, downsampling_factor=downsample_factor) #Generate downsampled HSI
upsized_image = np.repeat(np.repeat(lowres_downsampled, downsample_factor, axis=0), downsample_factor, axis=1)

spatial_transform_matrix = util.Gen_downsampled_spatial(downsample_factor,size).transpose() #Generate spatial transform for downsampling

rgb_mask = np.loadtxt("Calibration_data\\RGB_mask.txt")
spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)

rgb_representation = np.matmul(arr,spectral_response_matrix.T)

Upscaled_datacube, endmembers, abundances = CNMF(lowres_downsampled, 
                                                 rgb_representation, 
                                                 spatial_transform_matrix, 
                                                 spectral_response_matrix, 
                                                 VCA_init, 
                                                 endmember_count)


save_path = f"outputs\\{name}_{x_start}-{x_end}x_{y_start}-{y_end}y_{endmember_count}EM\\"

if not os.path.exists(save_path):
        os.mkdir(save_path)

save_final_image(arr, lowres_downsampled, Upscaled_datacube, spectral_response_matrix, save_path)
save_endmembers(endmembers, abundances, size, save_path)