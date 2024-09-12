import utilities as util
import numpy as np
import os

spatial_transform_matrix = util.Gen_downsampled_spatial(2,(50,50))

filename = f"Spatial_transform\\2_50x50_downsample.txt"
np.savetxt(X=spatial_transform_matrix.astype(np.float16),fname=filename, fmt="%.5f")