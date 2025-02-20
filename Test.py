from spatial_transform import get_distribution
import numpy as np
import matplotlib.pyplot as plt

img = np.memmap("Upscaled_cube.dat", dtype=np.float32, mode='r', shape=())
