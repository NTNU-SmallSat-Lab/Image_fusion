import numpy as np
from utilities import unflatten_datacube, save_RGB
import os

def save_HSI_as_RGB(Data, name, rgb):
    assert len(Data.shape) == 3, "Data array is not 3 dimensional"
    # Extract bands for R, G, B
    R = (Data[:, :, 0]*255).astype(np.uint8)
    G = (Data[:, :, 1]*255).astype(np.uint8)
    B = (Data[:, :, 2]*255).astype(np.uint8)

    rgb_image = np.stack((R, G, B), axis=-1)

    save_RGB(rgb_image, name)
    return