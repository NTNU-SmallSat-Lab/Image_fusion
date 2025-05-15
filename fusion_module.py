import numpy as np
from pathlib import Path
from Plotting import save_final_image, save_endmembers_many, save_endmembers_few, plot_final
import utilities as util
import os
from CNMF2 import CNMF
import loader as ld
import time
from CPPA import CPPA
from spatial_transform import spatial_transform
import json
import cv2
import matplotlib.pyplot as plt
from kernel import matrix_kernel_estimation_improved

class Fusion:
        def __init__(self, name = ""): #THIS IS still A BAD INIT FUNCTION
                if name == "":
                        self.data_string, self.name = util.Get_path("l1b file")
                else:
                        self.name = name
                        self.data_string = Path(f"C:/Users/phili/Desktop/Image_fusion/data/{self.name}.nc")
                self.read_config()
                self.flip = True
                self.patch_size = 100
                rgb_mask = np.loadtxt(self.rgb_mask)
                self.spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],108)*3.8
                self.loops = (self.inner_loops, self.outer_loops)
                self.load_images()
                self.force_overwrite = False #avoid regenerating transformed datacube
                repr = cv2.normalize(self.full_arr @ self.spectral_response_matrix.T, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite("uncorrected.png", repr)
                repr = cv2.normalize(util.remove_darkest(self.full_arr) @ self.spectral_response_matrix.T, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite("corrected.png", repr)
                print("done")
                os._exit(0)
                if self.remove_darkest:
                        self.full_arr = cv2.normalize(util.remove_darkest(self.full_arr), None, 0, 1, cv2.NORM_MINMAX)
                else:
                        self.full_arr = cv2.normalize(self.full_arr, None, 0, 1, cv2.NORM_MINMAX)
                self.spatial = spatial_transform(self.hsi_grayscale, self.rgb_grayscale)
                self.hsi_grayscale = self.spatial.hsi_image
                bands = 0
                self.full_arr = self.full_arr[self.spatial.hsi_limits[0]:self.spatial.hsi_limits[1], self.spatial.hsi_limits[2]:self.spatial.hsi_limits[3]]
                if not os.path.exists("Input_datacube.dat") or self.force_overwrite:
                        self.input_datacube = np.memmap("Input_datacube.dat", dtype=np.float32, mode='w+', shape=(self.rgb_grayscale.shape[0], self.rgb_grayscale.shape[1], 108))
                        for i in range(bands):
                                self.input_datacube[:,:,i] = cv2.warpPerspective(self.full_arr[:,:,i], #original HSI
                                                                                self.spatial.hr_transform, #transform from HSI to RGB coordinates
                                                                                (self.rgb_grayscale.shape[1], self.rgb_grayscale.shape[0])) #final resolution
                                self.input_datacube.flush()
                self.input_datacube = np.memmap("Input_datacube.dat", dtype=np.float32, mode='r', shape=(self.rgb_grayscale.shape[0], self.rgb_grayscale.shape[1], 108))
                self.full_arr = None #My poor RAM
        
        def load_images(self):
                # Normalize HSI Image
                normalized = cv2.normalize(ld.load_l1b_cube(self.data_string, bands=[6, 114])[250:450,:,:], None, 0, 1, cv2.NORM_MINMAX)
                cv2.imwrite("Raw_cube.png", cv2.normalize(normalized[:,:,::-1] @ self.spectral_response_matrix.T, None, 0, 255, cv2.NORM_MINMAX))
                
                self.endmember_list = np.reshape(normalized[::10,::10,:],shape=(-1,108)).T

                # Load RGB Image (Ensure RGB Conversion)
                rgb_path = str(self.data_string).replace("-", "_").replace("16Z_l1b.nc", "14.png")
                self.rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                if self.rgb_img is not None:  
                        self.rgb_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB)[1000:2800, 1800:2600,:]  # Convert BGR → RGB
                
                # Apply Flip and Rotate (Correct Order)
                if self.flip:
                        self.rgb_img = cv2.flip(self.rgb_img, 1)
                        self.rgb_img = cv2.rotate(self.rgb_img, cv2.ROTATE_90_CLOCKWISE)

                # Load Metadata
                meta_path = str(self.data_string).replace("l1b.nc", "meta.json")
                with open(meta_path, 'r') as file:
                        metadata = json.load(file)

                dx = metadata.get('gsd_along') / metadata.get('gsd_across')
                
                height, width = normalized.shape[:2]

                # Scale Image Properly
                if dx < 1:
                        scale_factor = 1.0 / dx
                        new_width = int(width * scale_factor)
                        new_height = height
                else:
                        scale_factor = dx
                        new_width = width
                        new_height = int(height * scale_factor)

                self.full_arr = cv2.resize(normalized, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

                # Compute HSI-RGB Mapping and Normalize
                hsi_rgb_float = cv2.normalize(self.full_arr[:,:,::-1] @ self.spectral_response_matrix.T, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite("Preprocessed_cube.png", hsi_rgb_float)
                hsi_rgb = np.clip(hsi_rgb_float, 0, 255).astype(np.uint8)

                # Convert to Grayscale (Ensure Correct Color Space)
                self.hsi_grayscale = cv2.cvtColor(hsi_rgb, cv2.COLOR_RGB2GRAY)
                self.rgb_grayscale = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2GRAY)
                
        def read_config(self):
                with open("config.txt", 'r') as file:
                        for line in file:
                                if line.strip() == "" or line.startswith("#"):
                                        continue

                                key, value = line.strip().split('=', 1)
                                key, value = key.strip(), value.strip()

                                if value.lower() in ['true', 'false']:
                                        setattr(self, key, value.lower() == 'true')
                                elif value.isdigit():
                                        setattr(self, key, int(value))
                                else:
                                        try:
                                                float_value = float(value)
                                                setattr(self, key, float_value)
                                        except ValueError:
                                                setattr(self, key, value)
        
        def fuse(self, HSI_patch, RGB_patch, spatial_transform):
                delta = self.delta
                if self.type == "PPA":
                        upscaled_patch, w, h = CPPA(HSI_data = HSI_patch, 
                                                  MSI_data= RGB_patch, 
                                                  spatial_transform= spatial_transform, 
                                                  spectral_response= self.spectral_response_matrix, 
                                                  endmember_list = self.endmember_list,
                                                  delta= delta, 
                                                  endmembers= self.endmember_n,
                                                  loops= self.loops,
                                                  tol= self.tol)
                        while np.abs(np.mean(np.sum(h, axis=0))-1) > 1e-1:
                                print(f"average abundance: {np.mean(np.sum(h, axis=0))} reducing delta from {delta}->{delta/2}")
                                delta = delta/2
                                upscaled_patch, w, h = CPPA(HSI_data = HSI_patch, 
                                                  MSI_data= RGB_patch, 
                                                  spatial_transform= spatial_transform, 
                                                  spectral_response= self.spectral_response_matrix, 
                                                  endmember_list = self.endmember_list,
                                                  delta= delta, 
                                                  endmembers= self.endmember_n,
                                                  loops= self.loops,
                                                  tol= self.tol)
                        save_endmembers_few(w, h, (self.patch_size, self.patch_size), f"patches/patch_{self.patch}")
                elif self.type == "CNMF":
                        CNMF_obj = CNMF(HSI_data = HSI_patch, 
                                              MSI_data= RGB_patch, 
                                              spatial_transform= spatial_transform, 
                                              spectral_response= self.spectral_response_matrix,
                                              delta= self.delta,
                                              endmembers= self.endmember_n,
                                              loops= self.loops,
                                              tol= self.tol)
                        upscaled_patch = CNMF_obj.final
                else:
                        raise ValueError(f"{self.type} is not a fusion method")
                """if np.abs(np.mean(np.sum(self.abundances, axis = 0))-1) > 1e-1:
                        self.delta = self.delta*0.5
                        print(f"Abundances outside of allowed values, reducing delta to {self.delta}")
                        self.fuse()"""
                return upscaled_patch
        
        def log_run(self):
                self.save_path = f"outputs\\{self.type}_{self.name}_{self.x_start}-{self.x_end}x_{self.y_start}-{self.y_end}y_{self.endmember_n}EM_{self.delta}d_{self.remove_darkest}RD\\"
                self.Result_values = {"Absolute mean error":self.mean_spatial_error,
                                 "Spectral_angle_Cosine":self.spectral_error,
                                 "Peak SNR":self.PSNR}
                self.Variable_values = {"Input":self.name,
                                   "Endmembers":self.endmember_n,
                                   "delta":self.delta,
                                   "loops":self.loops,
                                   "tolerance":self.tol,
                                   "Sigma":self.sigma,
                                   "Downsampling":self.downsample_factor,
                                   "Coordinates" : self.pix_coords, 
                                   "Runtime":self.runtime, 
                                   "Type": self.type, 
                                   "Remove_darkest": self.remove_darkest}
                util.log_results_to_csv("Runs.csv", variable_values=self.Variable_values, result_values=self.Result_values)
                self.save_files()
        
        def save_files(self):
                if not os.path.exists("outputs"):
                        os.mkdir("outputs")
                if not os.path.exists(self.save_path):
                        os.mkdir(self.save_path)
                save_final_image(Original= self.arr, 
                                 downscaled= self.lowres_downsampled, 
                                 Upscaled= self.Upscaled_datacube, 
                                 spectral_response_matrix= self.spectral_response_matrix, 
                                 save_path= self.save_path)
                if self.endmember_n > 10:
                        save_endmembers_many(self.endmembers, self.abundances, self.size, self.save_path)
                else:
                        save_endmembers_few(self.endmembers, self.abundances, self.size, self.save_path)
                results_path = f"{self.save_path}results.txt"
                with open(results_path, 'w') as file:
                        file.write("Result Values:\n")
                        for key, value in self.Result_values.items():
                                file.write(f"{key}: {value}\n")

                        file.write("\nVariable Values:\n")
                        for key, value in self.Variable_values.items():
                                file.write(f"{key}: {value}\n")
                                
        def fuse_image(self):
                start = time.time()
                final_cube_shape = (self.rgb_img.shape[0], self.rgb_img.shape[1], 108)
                self.upscaled_datacube = np.memmap("Upscaled_cube.dat", dtype=np.float32, mode='w+', shape=final_cube_shape)
                warped_grayscale = cv2.warpPerspective(self.hsi_grayscale, 
                                                       self.spatial.hr_transform,
                                                       (self.rgb_grayscale.shape[1], self.rgb_grayscale.shape[0]))
                self.blur_kernel = matrix_kernel_estimation_improved(warped_grayscale[200:400, 200:400], self.rgb_grayscale[200:400, 200:400], kernel_size=21)
                plt.imshow(self.blur_kernel)
                plt.show()
                os._exit(0)
                spatial_transform = self.gen_spatial(kernel_size=101, sigma=(0.01,10), rotation=5.0)
                done = False
                y = 1300
                while not done:
                        done_row = False
                        x = 600
                        while not done_row:
                                self.patch = f"{x}-{x+self.patch_size}_{y}-{y+self.patch_size}"
                                rgb_limits = np.array([x, x+self.patch_size, y, y+self.patch_size])
                                hsi_patch = self.input_datacube[rgb_limits[0]:rgb_limits[1], rgb_limits[2]: rgb_limits[3], :]
                                print(hsi_patch.shape)
                                hsi_data = hsi_patch.reshape(-1,hsi_patch.shape[2]).T
                                print(hsi_data.shape)
                                rgb_patch = self.rgb_img[rgb_limits[0]:rgb_limits[1], rgb_limits[2]:rgb_limits[3], :]
                                rgb_data = rgb_patch.reshape(-1, 3).T
                                upscaled_data = self.fuse(hsi_data, 
                                                        rgb_data, 
                                                        spatial_transform)
                                upscaled_patch = upscaled_data.T.reshape(rgb_patch.shape[0], rgb_patch.shape[1], hsi_patch.shape[2])
                                #upscaled_patch = cv2.normalize(upscaled_patch, None, 0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                self.upscaled_datacube[rgb_limits[0]:rgb_limits[1],
                                                        rgb_limits[2]:rgb_limits[3],:] = upscaled_patch
                                self.upscaled_datacube.flush()
                                x += self.patch_size
                                if x > self.rgb_img.shape[0] - self.patch_size:
                                        done_row = True
                                percent_done = y*100/(self.rgb_img.shape[1])+x*10000/(self.rgb_img.shape[0]*self.rgb_img.shape[1])
                                print(f"{percent_done:.2f}% completed")
                                break
                        y += self.patch_size
                        break
                        if y > self.rgb_img.shape[1] - self.patch_size:
                                done = True
                elapsed = time.time()-start
                print(f"Total run over in {elapsed} seconds, final datacube size is {self.upscaled_datacube.nbytes}\n final cube shape: {final_cube_shape}")
                cv2.imwrite("output/recent_band_30.png", (self.upscaled_datacube[:,:,27]*255).astype(np.uint8))
                cv2.imwrite("output/recent_band_50.png", (self.upscaled_datacube[:,:,47]*255).astype(np.uint8))
                cv2.imwrite("output/recent_band_70.png", (self.upscaled_datacube[:,:,67]*255).astype(np.uint8))
                cv2.imwrite("output/recent_rgb_base.png", cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite("output/recent_hsi_gray.png", self.hsi_grayscale)
                rgb_img = self.rgb_img[600:700,1300:1400,:]
                input_flattened = self.input_datacube[600:700,1300:1400, :].reshape(-1,hsi_patch.shape[2])
                input_RGB = np.reshape(input_flattened@self.spectral_response_matrix.T, shape=(100, 100, -1))
                output_flattened = self.upscaled_datacube[600:700,1300:1400, :].reshape(-1,hsi_patch.shape[2])
                output_RGB = np.reshape(output_flattened@self.spectral_response_matrix.T, shape=(100, 100, -1))
                plot_final(rgb_img, 
                           input_RGB, 
                           output_RGB,
                           spatial_transform)
                
        


        def gen_spatial(self, kernel_size=41, sigma=(1,1), rotation=0) -> np.ndarray: #Chat GPT
                """gaussian_x = cv2.getGaussianKernel(kernel_size, sigma=sigma[0])
                gaussian_y = cv2.getGaussianKernel(kernel_size, sigma=sigma[1])
                gaussian_2d = np.outer(gaussian_x, gaussian_y)
                if rotation != 0:
                        rMat = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), rotation, 1.0)
                        gaussian_2d = cv2.warpAffine(gaussian_2d, rMat, dsize=(kernel_size, kernel_size))"""
                
                patch_size = self.patch_size
                patch_spatial = np.zeros((patch_size * patch_size, patch_size * patch_size), dtype=np.float32)

                # Compute the kernel offsets
                x_kern, y_kern = np.meshgrid(np.arange(kernel_size) - kernel_size // 2,
                                                np.arange(kernel_size) - kernel_size // 2)
                x_kern = x_kern.ravel()
                y_kern = y_kern.ravel()

                # Compute all (x_0, y_0) positions
                x_0, y_0 = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
                x_0 = x_0.ravel()
                y_0 = y_0.ravel()

                # Compute the modified coordinates
                x_hsi = (x_0[:, None] + x_kern).flatten()
                y_hsi = (y_0[:, None] + y_kern).flatten()

                # Mask for valid coordinates
                valid_mask = (0 <= x_hsi) & (x_hsi < patch_size) & (0 <= y_hsi) & (y_hsi < patch_size)

                # Compute flat indices
                k_indices = np.repeat(np.arange(patch_size * patch_size), kernel_size * kernel_size)[valid_mask]
                hsi_indices = (x_hsi + y_hsi * patch_size)[valid_mask]

                # Assign Gaussian values efficiently
                patch_spatial[k_indices, hsi_indices] = np.tile(self.blur_kernel.ravel(), patch_size * patch_size)[valid_mask]
                assert patch_spatial.shape[0] == self.patch_size*self.patch_size and patch_spatial.shape[1] == self.patch_size*self.patch_size, "Spatial transform generation failed"
                return patch_spatial
        
        def get_blur_kernel(self, lmbda=50, beta=0.00001, kernel_size=75):
                rgb = cv2.imread('output/rgb.png')[::4,::4, 0]
                blurKernel = np.zeros(shape=(31,31))
                blurKernel[13:17,:] = np.ones_like(blurKernel[13:17,:])
                blurKernel[:,13:17] = np.ones_like(blurKernel[:,13:17])
                blurKernel /= np.sum(blurKernel)
                blurred = cv2.filter2D(rgb, ddepth=-1, kernel=blurKernel)
                
                
                plt.imshow(blurred)
                plt.show()

                os._exit(0)
                
                

                                
                                
def run(name):
    HSI_fusion = Fusion(name)
    HSI_fusion.fuse()
    HSI_fusion.log_run()
    ab_avg = np.abs(np.mean(np.sum(HSI_fusion.abundances, axis = 0)))
    
    # Prepare the line to append
    result_line = f"{name}, {HSI_fusion.delta}, {HSI_fusion.type}, {HSI_fusion.PSNR}, {HSI_fusion.spectral_error}, {HSI_fusion.runtime}, {ab_avg}\n"
    
    # Append to the results file
    with open("results", "a") as results_file:
        results_file.write(result_line)


if __name__ == "__main__":
        HSI_fusion = Fusion()
        HSI_fusion.fuse_image()
        