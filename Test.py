import numpy as np
import utilities as util
import loader as ld
import matplotlib.pyplot as plt
from Plotting import Normalize

# Load data path and cube
data_string, name = util.Get_path()
full_arr = ld.load_l1b_cube(data_string)

# Remove darkest pixels
darkest_removed = util.remove_darkest(full_arr)

# Load RGB mask and map to spectral bands
rgb_mask = np.loadtxt("RGB_mask.txt")
spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700, :], 112)

# Normalize and apply spectral response matrix
full_arr = Normalize(np.matmul(full_arr, spectral_response_matrix.T))
darkest_removed = Normalize(np.matmul(darkest_removed, spectral_response_matrix.T))

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Plot the full array in the first subplot
axes[0].imshow(full_arr)
axes[0].set_title("Full Array Image")
axes[0].axis('off')  # Hide axis for cleaner display

# Plot the darkest-removed array in the second subplot
axes[1].imshow(darkest_removed)
axes[1].set_title("Darkest Removed Image")
axes[1].axis('off')  # Hide axis for cleaner display

# Show the figure
plt.show()
