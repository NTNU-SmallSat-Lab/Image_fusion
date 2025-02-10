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
        def __init__(self, name = ""): #THIS IS A BAD INIT FUNCTION
                if name == "":
                        self.data_string, self.name = util.Get_path("l1b file")
                else:
                        self.name = name
                        self.data_string = Path(f"C:/Users/phili/Desktop/Image_fusion/data/{self.name}.nc")
                self.read_config()
                self.full_arr = ld.load_l1b_cube(self.data_string)
                rgb_path = str(self.data_string).replace("16Z-l1b.nc", "14.png")
                self.rgb_img = cv2.normalize(cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                meta_path = str(self.data_string).replace("l1a.nc", "meta.json")

                with open(meta_path, 'r') as file:
                        metadata = json.load(file)
                dx = metadata.get('gsd_along')/metadata.get('gsd_across')
                self.flip = True
                self.active_area, self.transform = full_transform(self.rgb_img, self.full_arr)
                self.full_arr = self.full_arr[self.active_area[0]:self.active_area[1],self.active_area[2]:self.active_area[3]]
                self.loops = (self.inner_loops, self.outer_loops)
                if self.remove_darkest:
                        self.arr = Normalize(util.remove_darkest(self.full_arr[self.x_start:self.x_end,self.y_start:self.y_end,:]), min=1E-6, max=1.0)
                        self.offset_arr = Normalize(util.remove_darkest(self.full_arr[self.x_start:self.x_end,self.y_start:self.y_end,:]), min=1E-6, max=1.0)
                else:
                        self.arr = Normalize(self.full_arr[self.x_start:self.x_end,self.y_start:self.y_end,:], min=1E-6, max=1.0)
                        self.offset_arr = Normalize(self.full_arr[self.x_start:self.x_end,self.y_start:self.y_end,:], min=1E-6, max=1.0)
                self.size = (self.active_area[1]-self.active_area[0],self.active_area[3]-self.active_area[2])
                if self.type == "CNMF":
                        self.w_init = Get_VCA(self.lowres_downsampled, self.endmember_n)
                        self.h_init = np.ones(shape=(self.endmember_n, self.lowres_downsampled.shape[0]*self.lowres_downsampled.shape[1]))
                self.spatial_transform_matrix = util.Gen_downsampled_spatial(self.downsample_factor,self.size).transpose()
                rgb_mask = np.loadtxt(self.rgb_mask_gen)
                self.spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)
                self.rgb_representation = np.matmul(self.arr,self.spectral_response_matrix.T)
                rgb_mask = np.loadtxt(self.rgb_mask)
                self.spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)
                        
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
        
        def fuse(self):
                start_time = time.time()
                if self.type == "PPA":
                        self.Upscaled_datacube, self.endmembers, self.abundances = ppa.CPPA(HSI_data = self.lowres_downsampled, 
                                                                                            MSI_data= self.rgb_representation, 
                                                                                            spatial_transform= self.spatial_transform_matrix, 
                                                                                            spectral_response= self.spectral_response_matrix, 
                                                                                            delta= self.delta, 
                                                                                            endmembers= self.endmember_n,
                                                                                            loops= self.loops,
                                                                                            tol= self.tol)
                elif self.type == "CNMF":
                        self.Upscaled_datacube, self.endmembers, self.abundances = CNMF(self.lowres_downsampled, 
                                                                                            self.rgb_representation, 
                                                                                            self.spatial_transform_matrix, 
                                                                                            self.spectral_response_matrix,
                                                                                            self.w_init, 
                                                                                            self.h_init,
                                                                                            self.delta, 
                                                                                            self.endmember_n,
                                                                                            self.loops,
                                                                                            self.tol)
                else:
                        raise ValueError(f"{self.type} is not a fusion method")
                if np.abs(np.mean(np.sum(self.abundances, axis = 0))-1) > 1e-1:
                        self.delta = self.delta*0.5
                        print(f"Abundances outside of allowed values, reducing delta to {self.delta}")
                        self.fuse()
                self.mean_spatial_error = np.mean(np.abs(self.arr - self.Upscaled_datacube))
                self.spectral_error = util.mean_spectral_angle(self.arr, self.Upscaled_datacube)
                self.PSNR = util.calculate_psnr(self.arr, self.Upscaled_datacube)
                self.runtime = time.time()-start_time
        
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
        HSI_fusion.fuse()
        HSI_fusion.log_run()