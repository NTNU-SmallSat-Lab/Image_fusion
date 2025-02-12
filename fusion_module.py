import numpy as np
from pathlib import Path
from Plotting import save_final_image, save_endmembers_many, Normalize, save_endmembers_few
import utilities as util
import os
from CNMF import CNMF, Get_VCA
import loader as ld
import time
import CPPA as ppa
from spatial_transform import full_transform, get_pixels
import json
import cv2

class Fusion:
        def __init__(self, name = ""): #THIS IS still A BAD INIT FUNCTION
                if name == "":
                        self.data_string, self.name = util.Get_path("l1b file")
                else:
                        self.name = name
                        self.data_string = Path(f"C:/Users/phili/Desktop/Image_fusion/data/{self.name}.nc")
                self.read_config()
                self.flip = True
                self.patch_size = 20
                rgb_mask = np.loadtxt(self.rgb_mask)
                self.spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)
                self.loops = (self.inner_loops, self.outer_loops)
                self.load_images()
                if self.remove_darkest:
                        self.arr = cv2.normalize(util.remove_darkest(self.full_arr), None, 0, 255, cv2.NORM_MINMAX)
                else:
                        self.arr = cv2.normalize(self.full_arr, None, 0, 255, cv2.NORM_MINMAX)
                self.get_transform()
                if self.type == "CNMF":
                        self.w_init = Get_VCA(self.lowres_downsampled, self.endmember_n)
                        self.h_init = np.ones(shape=(self.endmember_n, self.lowres_downsampled.shape[0]*self.lowres_downsampled.shape[1]))
        
        def load_images(self):
                normalized = cv2.normalize(ld.load_l1b_cube(self.data_string), None, 0, 255, cv2.NORM_MINMAX)
                normalized = np.uint8(normalized)  # Convert to uint8 for saving
                rgb_path = str(self.data_string).replace("-", "_")
                rgb_path = str(rgb_path).replace("16Z_l1b.nc", "14.png")
                self.rgb_img = np.uint8(cv2.normalize(cv2.imread(rgb_path, cv2.IMREAD_COLOR_RGB), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
                if self.flip:
                        self.rgb_img = cv2.flip(self.rgb_img,1)
                meta_path = str(self.data_string).replace("l1b.nc", "meta.json")

                with open(meta_path, 'r') as file:
                        metadata = json.load(file)
                dx = metadata.get('gsd_along')/metadata.get('gsd_across')
                
                height, width = normalized.shape[:2]

                if dx < 1:
                        scale_factor = 1.0 / dx
                        new_width = int(width * scale_factor)
                        new_height = height
                else:
                        scale_factor = dx
                        new_width = width
                        new_height = int(height * scale_factor)
                
                self.full_arr = cv2.resize(normalized, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                hsi_rgb = np.uint8(cv2.normalize(self.full_arr@self.spectral_response_matrix.T, None, 0, 255, cv2.NORM_MINMAX))
                self.hsi_grayscale = cv2.cvtColor(hsi_rgb, cv2.COLOR_RGB2GRAY)
                self.rgb_grayscale = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2GRAY)

                
                
        def get_transform(self):
                self.active_area, self.transform = full_transform(self.rgb_grayscale, self.hsi_grayscale)
                self.full_arr = self.full_arr[self.active_area[0]:self.active_area[1],self.active_area[2]:self.active_area[3]].copy()
                  
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
                        upscaled_patch, patch_endmembers, patch_abundances = ppa.CPPA(HSI_data = HSI_patch, 
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
                if np.abs(np.mean(np.sum(self.abundances, axis = 0))-1) > 1e-1:
                        self.delta = self.delta*0.5
                        print(f"Abundances outside of allowed values, reducing delta to {self.delta}")
                        self.fuse()
                return upscaled_patch, patch_endmembers, patch_abundances
        
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
                final_cube_shape = (self.active_area[1]-self.active_area[0], self.active_area[3]-self.active_area[2], 112)
                self.upscaled_datacube = np.memmap("Upscaled_cube.dat", dtype=np.uint8, mode='w+', shape=final_cube_shape)
                #~1.1 billion pixel values on upscaled cube
                done = False
                x, y = 20, 20
                while not done:
                        done_row = False
                        y_min = y
                        y_max = y+self.patch_size
                        while not done_row:
                                x_min = x
                                x_max = x+self.patch_size
                                limits = np.array([x_min, x_max, y_min, y_max])
                                hsi_patch = self.full_arr[x_min:x_max,y_min:y_max].copy()
                                spatial_transform, rgb_limits = get_pixels(limits, self.transform, self.rgb_img.shape)
                                rgb_patch = self.rgb_img[rgb_limits[0]:rgb_limits[1],rgb_limits[2]:rgb_limits[3],:]
                                upscaled_patch, patch_endmembers, patch_abundances = self.fuse(hsi_patch, rgb_patch, spatial_transform)
                                self.upscaled_datacube[x_min:x_max,y_min:y_max,:] = upscaled_patch
                                self.upscaled_datacube.flush()
                                x += self.patch_size
                                if x > self.full_arr.shape[0] - self.patch_size:
                                        done_row = True
                        y += self.patch_size
                        if y > self.full_arr.shape[1] - self.patch_size:
                                done = True
                elapsed = time.time()-start
                print(f"Total run over in {elapsed} seconds, final datacube size is {self.upscaled_datacube.nbytes}")
                                
                                
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