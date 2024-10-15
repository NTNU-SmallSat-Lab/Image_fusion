import numpy as np
from Plotting import save_final_image, save_endmembers_many, Normalize, save_endmembers_few
import utilities as util
import os
from CNMF import CNMF, Get_VCA
import loader as ld
import time
from Viewdata import visualize
from PPA import get_PPA
import PPA_NMF as ppa


data_string, name = util.Get_path()
start_time = time.time()
endmember_count = 3
delta = 0.15
tol = 0.00005
PPA = False
Unhaze = True
loops = (300, 5)
x_start, x_end, y_start, y_end = 100, 300, 0, 200

"""x_start = int(input("x_start: "))
x_end = int(input("x_end: "))
y_start = int(input("y_start"))
y_end = int(input("y_end"))"""

pix_coords = [x_start,x_end,y_start,y_end]

full_arr = ld.load_l1b_cube(data_string)

if Unhaze:
        arr = Normalize(util.remove_darkest(full_arr[x_start:x_end,y_start:y_end,:]), min=1E-6, max=1.0)
else:
        arr = Normalize(full_arr[x_start:x_end,y_start:y_end,:], min=1E-6, max=1.0)

size = (x_end-x_start,y_end-y_start)
downsample_factor = 4
sigma = 4

lowres_downsampled = util.Downsample(arr, sigma=sigma, downsampling_factor=downsample_factor) #Generate downsampled HSI

if PPA:
        w_init, h_init = get_PPA(lowres_downsampled, endmember_count, delta=delta)
else:
        w_init, h_init = Get_VCA(lowres_downsampled, endmember_count), np.ones(shape=(endmember_count, lowres_downsampled.shape[0]*lowres_downsampled.shape[1]))


upsized_image = np.repeat(np.repeat(lowres_downsampled, downsample_factor, axis=0), downsample_factor, axis=1)

spatial_transform_matrix = util.Gen_downsampled_spatial(downsample_factor,size).transpose() #Generate spatial transform for downsampling

rgb_mask = np.loadtxt("RGB_mask.txt")
spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)

rgb_representation = np.matmul(arr,spectral_response_matrix.T)

"""Upscaled_datacube, endmembers, abundances = CNMF(lowres_downsampled, 
                                                 rgb_representation, 
                                                 spatial_transform_matrix, 
                                                 spectral_response_matrix, 
                                                 w_init=w_init,
                                                 h_init=h_init,
                                                 endmembers=endmember_count,
                                                 delta=delta,
                                                 loops=loops,
                                                 tol=tol)"""

"""while np.mean(np.sum(abundances, axis=0)) - 1 > 1E-4:
        delta = 0.75*delta
        print(f"Abundances outside constraints, reducing delta to {delta} and rerunning")
        Upscaled_datacube, endmembers, abundances = CNMF(lowres_downsampled, 
                                                 rgb_representation, 
                                                 spatial_transform_matrix, 
                                                 spectral_response_matrix, 
                                                 w_init=w_init,
                                                 h_init=h_init,
                                                 endmembers=endmember_count,
                                                 delta=delta,
                                                 loops=loops,
                                                 tol=tol)"""

Upscaled_datacube, endmembers, abundances = ppa.CPPA(lowres_downsampled, rgb_representation, spatial_transform_matrix, spectral_response_matrix, delta, endmember_count)

abundances = np.clip(abundances, a_min=0, a_max=1)        
"""endmembers, abundances = get_PPA(Upscaled_datacube, endmember_count, delta, endmembers, abundances)
Upscaled_datacube = np.matmul(endmembers, abundances).T.reshape(Upscaled_datacube.shape[0], 
                                                                          Upscaled_datacube.shape[1], 
                                                                          Upscaled_datacube.shape[2])"""
        


if PPA:
        type = "PPA"
else:
        type = "VCA"

save_path = f"outputs\\{type}_{name}_{x_start}-{x_end}x_{y_start}-{y_end}y_{endmember_count}EM_{delta}d\\"

if not os.path.exists(save_path):
        os.mkdir(save_path)

print(f"Mean abundance per pixel sum: {np.mean(np.sum(abundances, axis=0))}")
#assert abs(np.mean(np.sum(abundances, axis=0))-1) < 0.01, "Abundances do not sum to 1, reduce delta or increase out loop number"

save_final_image(arr, lowres_downsampled, Upscaled_datacube, spectral_response_matrix, save_path)
if endmember_count > 10:
        save_endmembers_many(endmembers, abundances, size, save_path)
else:
        save_endmembers_few(endmembers, abundances, size, save_path)

mean_spatial_error = np.mean(np.abs(arr - Upscaled_datacube))
spectral_error = util.mean_spectral_angle(arr, Upscaled_datacube)
PSNR = util.calculate_psnr(arr, Upscaled_datacube)
Result_values = {"Absolute mean error":mean_spatial_error,"Spectral_angle_Cosine":spectral_error,"Peak SNR":PSNR}
end_time = time.time()
Variable_values = {"Input":name,"Endmembers":endmember_count,"delta":delta,"loops":loops,"tolerance":tol,"Sigma":sigma,"Downsampling":downsample_factor,"Coordinates" : pix_coords, "Runtime":(end_time-start_time), "PPA": PPA, "Unhaze": Unhaze}
util.log_results_to_csv("Runs.csv", variable_values=Variable_values, result_values=Result_values)

print(f"Saved in {save_path}")

#visualize(endmembers, abundances, size)