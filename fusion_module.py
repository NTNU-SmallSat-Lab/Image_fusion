import numpy as np
from pathlib import Path
from Plotting import save_final_image, save_endmembers_many, Normalize, save_endmembers_few
import utilities as util
import os
from CNMF import CNMF, Get_VCA
import loader as ld
import time
import CPPA as ppa
from spatial_transform import spatial_transform, remove_unused, rebuild_data
import json
import cv2
import matplotlib.pyplot as plt

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
                self.spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)
                self.loops = (self.inner_loops, self.outer_loops)
                self.load_images()
                if self.remove_darkest:
                        self.arr = cv2.normalize(util.remove_darkest(self.full_arr), None, 0, 1, cv2.NORM_MINMAX)
                else:
                        self.arr = cv2.normalize(self.full_arr, None, 0, 1, cv2.NORM_MINMAX)
                self.spatial = spatial_transform(self.hsi_grayscale, self.rgb_grayscale)
                if self.type == "CNMF":
                        self.w_init = Get_VCA(self.lowres_downsampled, self.endmember_n)
                        self.h_init = np.ones(shape=(self.endmember_n, self.lowres_downsampled.shape[0]*self.lowres_downsampled.shape[1]))
        
        def load_images(self):
                # Normalize HSI Image
                normalized = cv2.normalize(ld.load_l1b_cube(self.data_string)[150:450,:,:], None, 0, 1, cv2.NORM_MINMAX)

                # Load RGB Image (Ensure RGB Conversion)
                rgb_path = str(self.data_string).replace("-", "_").replace("16Z_l1b.nc", "14.png")
                self.rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)[1000:3000,1000:2500]
                if self.rgb_img is not None:  
                        self.rgb_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
                
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
                hsi_rgb_float = cv2.normalize(self.full_arr @ self.spectral_response_matrix.T, None, 0, 255, cv2.NORM_MINMAX)
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
                if self.type == "PPA":
                        upscaled_patch = ppa.CPPA(HSI_data = HSI_patch, 
                                                  MSI_data= RGB_patch, 
                                                  spatial_transform= spatial_transform, 
                                                  spectral_response= self.spectral_response_matrix, 
                                                  delta= self.delta, 
                                                  endmembers= self.endmember_n,
                                                  loops= self.loops,
                                                  tol= self.tol)
                elif self.type == "CNMF":
                        upscaled_patch, patch_endmembers, patch_abundances = CNMF(HSI_data = HSI_patch, 
                                                                                            MSI_data= RGB_patch, 
                                                                                            spatial_transform= spatial_transform, 
                                                                                            spectral_response= self.spectral_response_matrix,
                                                                                            w_init=self.w_init, 
                                                                                            h_init=self.h_init,
                                                                                            delta= self.delta, 
                                                                                            endmembers= self.endmember_n,
                                                                                            loops= self.loops,
                                                                                            tol= self.tol)
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
                final_cube_shape = (self.rgb_img.shape[1], self.rgb_img.shape[0], 112)
                self.upscaled_datacube = np.memmap("Upscaled_cube.dat", dtype=np.float32, mode='w+', shape=final_cube_shape)
                #~1.1E9 pixel values on upscaled cube
                done = False
                y = 0
                while not done:
                        done_row = False
                        x = 0
                        while not done_row:
                                rgb_limits = np.array([x, x+self.patch_size, y, y+self.patch_size])
                                spatial_transform, hsi_limits =  self.spatial.get_pixels2(rgb_limits)
                                if spatial_transform is None or hsi_limits is None:
                                        print(f"Skipped patch ({x}:{x+self.patch_size},{y}:{y+self.patch_size}) due to getpixel->None")
                                        x += self.patch_size
                                        if x > self.rgb_img.shape[1] - self.patch_size:
                                                done_row = True
                                        continue
                                hsi_patch = self.full_arr[hsi_limits[0]:hsi_limits[1], hsi_limits[2]: hsi_limits[3], :]
                                hsi_data = hsi_patch.reshape(-1,hsi_patch.shape[2]).T
                                rgb_patch = self.rgb_img[rgb_limits[0]:rgb_limits[1], rgb_limits[2]:rgb_limits[3], :]
                                rgb_data = rgb_patch.reshape(-1, 3).T
                                pruned_transform, pruned_hsi_data = remove_unused(spatial_transform, hsi_data)
                                if pruned_hsi_data.shape[1] < 5:
                                        print(f"Skipped patch ({x}:{x+self.patch_size},{y}:{y+self.patch_size}) due to 5>hsi_pixels")
                                        x += self.patch_size
                                        if x > self.rgb_img.shape[1] - self.patch_size:
                                                done_row = True
                                        continue
                                try:
                                        upscaled_data = self.fuse(pruned_hsi_data, 
                                                                rgb_data, 
                                                                pruned_transform.T)
                                except:
                                        print(f"Skipped patch ({x}:{x+self.patch_size},{y}:{y+self.patch_size}) due to fusion failure")
                                        x += self.patch_size
                                        if x > self.rgb_img.shape[1] - self.patch_size:
                                                done_row = True
                                        continue
                                #upscaled_patch = rebuild_data(upscaled_data,rgb_mask)
                                upscaled_patch = upscaled_data.T.reshape(rgb_patch.shape[0], rgb_patch.shape[1], hsi_patch.shape[2])
                                upscaled_patch = cv2.normalize(upscaled_patch, None, 0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                                self.upscaled_datacube[rgb_limits[0]:rgb_limits[1],
                                                        rgb_limits[2]:rgb_limits[3],:] = upscaled_patch
                                self.upscaled_datacube.flush()
                                #print(f"Done patch ({x}:{x+self.patch_size},{y}:{y+self.patch_size})")
                                x += self.patch_size
                                if x > self.rgb_img.shape[1] - self.patch_size:
                                        done_row = True
                        y += self.patch_size
                        percent_done = y*100/(self.rgb_img.shape[0])
                        print(f"{percent_done}% completed")
                        if y > self.rgb_img.shape[0] - self.patch_size:
                                done = True
                elapsed = time.time()-start
                print(f"Total run over in {elapsed} seconds, final datacube size is {self.upscaled_datacube.nbytes}\n final cube shape: {final_cube_shape}")
                plt.imshow(self.upscaled_datacube[:,:,50])
                plt.show()
                                
                                
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