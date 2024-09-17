import numpy as np
from saver import save_HSI_as_RGB
import utilities as util
import os
from CNMF import CNMF, Get_VCA
import loader as ld
import matplotlib.pyplot as plt


data_string, name = util.Get_path()

if not os.path.exists(f"output_images\\{name}"):
        os.mkdir(f"output_images\\{name}")
precision = np.float64

rgb = {
    "R": 72,
    "G": 43,
    "B": 15
}

endmember_count = 40

pix_coords = [0,304,0,304]
VCA_init = Get_VCA(data_string, endmember_count)

full_arr = util.normalize(ld.load_l1b_cube(data_string))

arr = full_arr[pix_coords[0]:pix_coords[1],pix_coords[2]:pix_coords[3],:]
bands = full_arr.shape[2]
size = (pix_coords[1]-pix_coords[0],pix_coords[3]-pix_coords[2])
downsample_factor = 4
sigma = 1

rgb_representation = arr[:,:,[rgb["R"],rgb["G"],rgb["B"]]] #Generate RGB representation of original MSI

lowres_downsampled = util.Downsample(arr, sigma=sigma, downsampling_factor=downsample_factor) #Generate downsampled HSI
upsized_image = np.repeat(np.repeat(lowres_downsampled, downsample_factor, axis=0), downsample_factor, axis=1)

spatial_transform_matrix = util.Gen_downsampled_spatial(downsample_factor,size).transpose() #Generate spatial transform for downsampling

spectral_response_matrix = util.Gen_spectral(rgb=rgb, bands=bands, spectral_spread=3)

Upscaled_datacube = CNMF(lowres_downsampled, rgb_representation, spatial_transform_matrix, spectral_response_matrix, VCA_init, endmember_count)

error = util.get_error(Upscaled_datacube,arr)
error_rgb = np.stack([error] * 3, axis=-1)

top = np.hstack([arr[:,:,[72, 43, 15]],upsized_image[:,:,[72, 43, 15]]])
bottom = np.hstack([Upscaled_datacube[:,:,[72, 43, 15]],error_rgb])
final_image = np.vstack([top,bottom])

save_HSI_as_RGB(final_image, name=f"{name}\\Final_{downsample_factor}DS_{endmember_count}em.png", rgb=rgb)

spec_error = util.get_spectral_error(Upscaled_datacube, arr)
plt.figure(figsize=(8,4))
plt.plot(spec_error, linestyle='-', label='Spectral error sum')
plt.vlines(x=[72, 43, 15], ymin=0, ymax=spec_error.max(), colors=['r', 'g', 'b'])
plt.title('Spectral error')
plt.xlabel('bands')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()