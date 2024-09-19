import numpy as np
from PIL import Image

img = Image.open("Output_filled.PNG")

data = np.asarray(img, dtype=np.uint8)

data = data[:,:,0:3]

start_nm = 350
end_nm = 700
start_level = 0.0
end_level = 0.5

level = np.zeros(shape=(data.shape[0],4),dtype=np.float64)

for i in range(data.shape[0]):
    level[i,0] = 350+350*i/data.shape[1]
    for j in range(data.shape[1]):
        if data[i,j, 0] == 255:
            level[i, 1] = 0.5*j/data.shape[0]
        if data[i,j, 1] == 255:
            level[i, 2] = 0.5*j/data.shape[0]
        if data[i,j, 2] == 255:
            level[i, 3] = 0.5*j/data.shape[0]

np.savetxt(x=level, fname="RGB_levels")