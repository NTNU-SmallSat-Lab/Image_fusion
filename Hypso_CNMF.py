import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from VCA_master.VCA import vca
import loader as ld
from saver import save_HSI_as_RGB
import utilities as util
import os
from CNMF import CNMF

#data_string, name = util.Get_path()
data_string, name = "C:\\Users\\philipdb\\Desktop\\Project_work\\code\\Non-Negative_Matrix_factorization\\hypso_1_datacubes\\mjosa_2023-04-22_0932Z-l1a_mini.txt", "mjosa_2023-04-22_0932Z-l1a_mini"
if not os.path.exists(f"output_images\\{name}"):
        os.mkdir(f"output_images\\{name}")
precision = np.float64

rgb = {
    "R": 75,
    "G": 46,
    "B": 18
}
original_size = (200,200,120)
#original_size = (598, 1092, 120)
#original_size = (956, 684, 120) #TODO implement wide/nominal check and autoset size
pix_coords = [0,200,0,200]

arr = util.Get_subset(data_string, original_size, pix_coords)
bands = arr.shape[2]
arr = util.normalize(arr.astype(precision)) #Unsure what sort of precision is required here
size = (pix_coords[1]-pix_coords[0],pix_coords[3]-pix_coords[2])
downsample_factor = 2
sigma = 1

save_HSI_as_RGB(arr, name=f"{name}\\original.png", rgb=rgb) #Save original subset GROUND TRUTH

rgb_representation = arr[:,:,[rgb["R"],rgb["G"],rgb["B"]]] #Generate RGB representation of original MSI
lowres_downsampled = util.Downsample(arr, sigma=sigma, downsampling_factor=downsample_factor) #Generate downsampled HSI

save_HSI_as_RGB(lowres_downsampled, name=f"{name}\\Downscaled.png", rgb=rgb) #Save HSI

endmember_count = 30

spatial_transform_matrix = util.Gen_downsampled_spatial(downsample_factor,size).transpose() #Generate spatial transform for downsampling

spectral_response_matrix = util.Gen_spectral(rgb=rgb, bands=bands, spectral_spread=5)

Upscaled_datacube = CNMF(lowres_downsampled, rgb_representation, spatial_transform_matrix, spectral_response_matrix, endmember_count)

save_HSI_as_RGB(Upscaled_datacube, name=f"{name}\\Output.png", rgb=rgb)