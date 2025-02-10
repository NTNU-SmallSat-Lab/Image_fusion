import scipy.sparse as sparse
import numpy as np

array = np.load(r"C:\Users\phili\Desktop\Image_fusion\hypso2_data\h2_radiometric_calibration_matrix_wide.npz")
print(array["arr_0"].shape)