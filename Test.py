import utilities as util
import numpy as np
import netCDF4 as nc
import loader as ld

rgb = [72, 43, 15]

bands = 120
spectral_response_matrix = util.Gen_spectral(rgb=rgb, bands=bands, spectral_spread=1)

multiply = np.stack([spectral_response_matrix[0]]*200, axis=-1)
multiply = np.stack([multiply]*200, axis=-1)
multiply = multiply.transpose(1,2,0)

print(multiply.shape)