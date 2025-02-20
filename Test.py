import numpy as np
import matplotlib.pyplot as plt

file_path = "Upscaled_cube.dat"
dtype = np.float32

# Shape (Swapping width & height)
shape = (1500, 1748, 112)  # TRY SWITCHING width & height

# Read slice 70
slice_index = 70
img = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)[:, :, slice_index]

# Display the image
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.show()
