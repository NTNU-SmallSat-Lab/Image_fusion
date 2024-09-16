import numpy as np
from saver import save_HSI_as_RGB
import utilities as util
import os
from CNMF import CNMF, Get_VCA
import loader as ld


data_string, name = util.Get_path()

if not os.path.exists(f"output_images\\{name}"):
        os.mkdir(f"output_images\\{name}")
precision = np.float64

rgb = {
    "R": 75,
    "G": 46,
    "B": 18
}

endmember_count = 40

original_size = ld.load_l1b_shape(data_string)

pix_coords = [0,400,0,400]
VCA_init = Get_VCA(data_string, endmember_count)

full_arr = util.normalize(ld.load_l1b_cube(data_string))

arr = full_arr[pix_coords[0]:pix_coords[1],pix_coords[2]:pix_coords[3],:]
bands = full_arr.shape[2]
size = (pix_coords[1]-pix_coords[0],pix_coords[3]-pix_coords[2])
downsample_factor = 16

sigma = 1

#save_HSI_as_RGB(arr, name=f"{name}\\original_{downsample_factor}DS_{endmember_count}em.png", rgb=rgb) #Save original subset GROUND TRUTH

rgb_representation = arr[:,:,[rgb["R"],rgb["G"],rgb["B"]]] #Generate RGB representation of original MSI
lowres_downsampled = util.Downsample(arr, sigma=sigma, downsampling_factor=downsample_factor) #Generate downsampled HSI
upsized_image = np.repeat(np.repeat(lowres_downsampled, downsample_factor, axis=0), downsample_factor, axis=1)

#save_HSI_as_RGB(upsized_image, name=f"{name}\\Downscaled_{downsample_factor}DS.png", rgb=rgb) #Save HSI

spatial_transform_matrix = util.Gen_downsampled_spatial(downsample_factor,size).transpose() #Generate spatial transform for downsampling

spectral_response_matrix = util.Gen_spectral(rgb=rgb, bands=bands, spectral_spread=3)

Upscaled_datacube = CNMF(lowres_downsampled, rgb_representation, spatial_transform_matrix, spectral_response_matrix, VCA_init, endmember_count)
error = util.get_error(Upscaled_datacube,arr)
#util.save_grayscale_image(error, name=f"{name}\\Error_{downsample_factor}DS_{endmember_count}em.png")
rgb = [75, 46, 18]
error_rgb = np.stack([error] * 3, axis=-1)
top = np.hstack([arr[:,:,rgb],upsized_image[:,:,rgb]])
bottom = np.hstack([Upscaled_datacube[:,:,rgb],error_rgb])
final_image = np.vstack([top,bottom])
rgb = {
    "R": 75,
    "G": 46,
    "B": 18
}
save_HSI_as_RGB(final_image, name=f"{name}\\Final_{downsample_factor}DS_{endmember_count}em.png", rgb=rgb)

#save_HSI_as_RGB(Upscaled_datacube, name=f"{name}\\Output_{downsample_factor}DS_{endmember_count}em.png", rgb=rgb)