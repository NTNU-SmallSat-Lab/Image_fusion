import numpy as np
import matplotlib.pyplot as plt
from utilities import Get_path
from loader import load_l1b_cube

# Define the shape of the data cube
shape = (800, 1748, 108)
Upscaled_path = "Upscaled_cube.dat"
input_path = "Input_datacube.dat"
dtype = np.float32

start_x, start_y = 600, 1300
patch_size = 100

upscaled_cube = np.memmap(Upscaled_path, dtype, 'r', shape=shape)
original_cube = np.memmap(input_path, dtype, 'r', shape=shape)

# Define points to extract spectra from
points = [(20, 20), (20, 40), (20, 60), (20, 80)]

# Extract spectra for each point
spectra_upscaled = [upscaled_cube[x+start_x, y+start_y, :] for x, y in points]
spectra_original = [original_cube[x+start_x, y+start_y, :] for x, y in points]

# Extract the slice at band index 50
band_index = 50
upscaled_image = upscaled_cube[start_x:start_x+patch_size, start_y:start_y+patch_size, band_index]
original_image = original_cube[start_x:start_x+patch_size, start_y:start_y+patch_size, band_index]
# Create the figure with two subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the spectra
for idx, spec in enumerate(spectra_upscaled):
    line, = axes[0, 0].plot(spec, label=f"Point {idx+1}")

axes[0, 0].set_xlabel("Band Index")
axes[0, 0].set_ylabel("Value")
axes[0, 0].set_title("Spectral Profiles")
axes[0, 0].legend()

# Show the band image
im = axes[1, 0].imshow(upscaled_image, cmap="gray")
axes[1, 0].set_title(f"Band {band_index} Upscaled")
axes[1, 0].axis("off")

# Overlay points in corresponding colors
colors = [line.get_color() for line in axes[0, 0].lines]
for idx, (x, y) in enumerate(points):
    axes[1, 0].scatter(y, x, color=colors[idx], edgecolor="white", s=50, label=f"Point {idx+1}")
    
# Plot the spectra
for idx, spec in enumerate(spectra_original):
    line, = axes[0, 1].plot(spec, label=f"Point {idx+1}")

axes[0, 1].set_xlabel("Band Index")
axes[0, 1].set_ylabel("Value")
axes[0, 1].set_title("Spectral Profiles")
axes[0, 1].legend()

# Show the band image
im = axes[1, 1].imshow(original_image, cmap="gray")
axes[1, 1].set_title(f"Band {band_index} original")
axes[1, 1].axis("off")

# Overlay points in corresponding colors
colors = [line.get_color() for line in axes[0, 0].lines]
for idx, (x, y) in enumerate(points):
    axes[1, 1].scatter(y, x, color=colors[idx], edgecolor="white", s=50, label=f"Point {idx+1}")

plt.show()