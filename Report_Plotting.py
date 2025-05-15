import numpy as np
import cv2
import os
import cvxpy as cp
import sys
import utilities as util
import matplotlib.pyplot as plt
from spatial_transform import spatial_transform
save_string = "report\\"

"""hsi = np.load(f"{save_string}transformed_normalized_darkest_removed.npy", mmap_mode='r')#[180:2595:100,860::100].astype(np.float16)
hsi = hsi[180:2595,860:].astype(np.float32)
cube = np.load(f"{save_string}normalized_cropped.npy", mmap_mode='r')
cube = util.remove_darkest(cube[:,:,1:])
hsi = np.reshape(hsi, shape=(hsi.shape[0]*hsi.shape[1], -1))
rgb = cv2.flip(np.load(f"{save_string}rgb_darkest_removed_RGB_normalized.npy").astype(np.float16), 1)[180:2595:200,860::200]
rgb_r = rgb[:,:,1].reshape(rgb.shape[0]*rgb.shape[1],-1)
print(hsi.shape,rgb_r.shape)
pixels, bands = hsi.shape

X = cp.Variable(bands)

objective = cp.Minimize(0.5*cp.sum_squares(hsi@X-rgb_r))

constraint = [X >= 0, cp.sum(X) == 1]

prob = cp.Problem(objective, constraint)

prob.solve(verbose=True)

print(X.value)"""

"""rgb_capture_cropped = cv2.flip(np.load(f"{save_string}rgb_darkest_removed_RGB_normalized.npy").astype(np.float32), 1)[224:2615,:3814,:]
np.save(f"{save_string}RGB_CAPTURE_CROPPED", rgb_capture_cropped)
#slice = cv2.normalize(raw_cube[:,:,60], None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
#spatial = spatial_transform(slice, cv2.normalize(rgb_capture[:,:,1], None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8))

input_datacube = np.load(f"{save_string}transformed_raw_cube.npy", mmap_mode='r')[224:2615,:3814,:]
np.save(f"{save_string}HSI_RAW_TRANSFORMED_CROPPED", input_datacube)"""

#SHOW KERNEL
spectral_transform = np.load(f"{save_string}Bodged_spectral_transform.npy")
spectral_transform_orig = np.load(f"{save_string}original_spectral_transform.npy")
#cube = np.load(f"{save_string}normalized_cropped.npy", mmap_mode='r')
rgb = np.load(f"{save_string}RGB_CAPTURE_CROPPED.npy").astype(np.float32)
hsi = np.load(f"{save_string}HSI_RAW_TRANSFORMED_CROPPED.npy", mmap_mode='r')
#hsi = cv2.normalize(hsi, None, 0, 1.0, norm_type=cv2.NORM_MINMAX) #REMOVE FIRST BAND -> NORMALIZE -> FIX SPECTRAL TRANSFORM
hsi_rgb = cv2.normalize(hsi@(spectral_transform[:,1:-1].astype(np.float32).T), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
print(np.max(rgb))
diff = rgb-hsi_rgb
print(np.sum(diff, axis=(0, 1))/(rgb.shape[0]*rgb.shape[1]))

print(np.min(hsi[:,:,1:]), np.max(hsi[:,:,1:]))

kernel_size=55
sigma = (3, 50)
rotation = 5

gaussian_x = cv2.getGaussianKernel(kernel_size, sigma=sigma[0])
gaussian_y = cv2.getGaussianKernel(kernel_size, sigma=sigma[1])
gaussian_2d = np.outer(gaussian_x, gaussian_y)
gaussian_2d /= np.sum(gaussian_2d)
if rotation != 0:
        rMat = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), rotation, 1.0)
        gaussian_2d = cv2.warpAffine(gaussian_2d, rMat, dsize=(kernel_size, kernel_size))
rgb_blurred = cv2.filter2D(rgb, -1, gaussian_2d)
print(f"Sizes {rgb.shape, hsi_rgb.shape, rgb_blurred.shape}")
plt.figure(figsize=(12, 4))
for i, (img, title) in enumerate(zip([rgb, hsi_rgb, rgb_blurred], ['RGB', 'HSI', 'RGB Blurred'])):
    plt.subplot(1, 3, i+1); plt.imshow(img); plt.title(title); plt.axis('off')
plt.tight_layout(); plt.show()