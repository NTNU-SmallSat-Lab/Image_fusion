import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from utilities import Get_path

file_path = "Upscaled_cube.dat"
dtype = np.float32

shape = (800, 1748, 112)

x, y = 341, 1240  # Indices of the slice to extract

# Calculate the byte offset
element_size = np.dtype(dtype).itemsize  # 4 bytes for np.float32
row_size = shape[1] * shape[2]  # Elements in a row
offset = (x * row_size + y * shape[2]) * element_size
img = np.memmap(file_path, dtype=dtype, mode='r', shape=(shape[2],), offset=offset)

# Display the image
#plt.imshow(img)#, cmap='gray')
plt.plot(img)
#plt.colorbar()
plt.show()