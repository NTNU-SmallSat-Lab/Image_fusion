import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
import pandas as pd
import subprocess
import csv
from datetime import datetime

def Cost(Data1, Data2, band1=0, band2=0):
        """calculates the squared frobenius norm between two np.arrays

        Args:
            Data1 (np.array): flattened datacube (b,xy)
            Data2 (np.array): flattened datacube (b,xy)
            band1 (int, optional): start band of cost. Defaults to 0.
            band2 (int, optional): end band of cost. Defaults to 0.
        if no bands are specified calculates all bands
        Returns:
            float32: scalar cost
        """
        if band2 == 0:
                band2 = Data1.shape[0]
        assert Data1.shape == Data2.shape, "Data not same dimensions"
        cost = np.linalg.norm(Data1[band1:band2,:]-Data2[band1:band2,:],ord="fro")**2
        return cost

def Wavelength_to_band(wavelength):
    wavelengths = np.asarray(pd.read_csv("Calibration_data\\spectral_bands_Hypso-1_v1.csv"))[:,0]
    for i in range(wavelengths.shape[0]):
         if (wavelengths[i] < wavelength) and (wavelengths[i+1]>wavelength):
              return i
    return -1

def band_to_wavelength(band: int):
    wavelengths = np.asarray(pd.read_csv("Calibration_data\\spectral_bands_Hypso-1_v1.csv"))[:,0]
    return (wavelengths[band],wavelengths[band+1])
    

def Get_path():
        # Initialize Tkinter and hide the main window
        root = Tk()
        root.withdraw()

        # Open the file explorer and allow the user to select a file
        file_path_input = askopenfilename(title="Select a file", initialdir=Path.cwd())

        # Initialize a Path object
        file_path = Path(file_path_input)
        name = file_path.stem
        return file_path, name

def Get_subset(path, size, subset) -> np.array:
        flattened_array = np.loadtxt(fname=path)
        array = flattened_array.reshape(size)
        subset_array = array[subset[0]:subset[1],subset[2]:subset[3],:]
        return subset_array

def save_RGB(data, name):
    assert data.shape[2] == 3, "Colour channels != 3"
    path = f"output_images\\{name}"
    img = Image.fromarray(data)
    img.save(path)
    return

def Downsample(data: np.array, sigma=1, downsampling_factor=2) -> np.array:
    assert (data.shape[0]%downsampling_factor == 0) and (data.shape[1]%downsampling_factor == 0), "Resolution must be whole multiple of downsample factor"
    blurred = gaussian_filter(data, sigma=(sigma, sigma, 0), mode='reflect')
    lowres_downsampled = np.zeros(shape=(int(data.shape[0]/downsampling_factor),int(data.shape[1]/downsampling_factor),data.shape[2]))
    lowres_downsampled[:,:,:] = blurred[::downsampling_factor,::downsampling_factor,:]
    return lowres_downsampled

def Gen_downsampled_spatial(downsampling_factor, size) -> np.array:
    reduced_size = (int(size[0]/downsampling_factor),int(size[1]/downsampling_factor))
    high_res, low_res = size[0]*size[1], reduced_size[0]*reduced_size[1]
    spatial_transform = np.zeros(shape=(low_res,high_res))
    for i in range(reduced_size[0]):
        for j in range(reduced_size[1]):
             for k in range(downsampling_factor):
                  spat_x = (i*reduced_size[0]+j)
                  spat_y = (i*size[0]+j)*downsampling_factor+k*downsampling_factor*reduced_size[0]
                  spatial_transform[spat_x,spat_y:spat_y+downsampling_factor] = np.ones(shape=(1,downsampling_factor))/(downsampling_factor**2)
            
    return spatial_transform.astype(np.float64)

def Gen_spectral(rgb, bands, spectral_spread) -> np.array:
    spectral_response_matrix = np.zeros(shape=(3,bands))
    spectral_response_matrix[0,rgb[0]-spectral_spread:rgb[0]+spectral_spread+1] = 1
    spectral_response_matrix[1,rgb[1]-spectral_spread:rgb[1]+spectral_spread+1] = 1
    spectral_response_matrix[2,rgb[2]-spectral_spread:rgb[2]+spectral_spread+1] = 1 #sum up a few bands around each RGB component since no calibration data
    spectral_response_matrix = spectral_response_matrix/spectral_response_matrix.sum(axis=1, keepdims=True)
    return spectral_response_matrix

def map_mask_to_bands(mask: np.array, bands: int):
    output = np.zeros(shape=(mask.shape[1],bands))
    for i in range(mask.shape[0]):
        band = Wavelength_to_band(i)
        if band != -1:
            output[:,band] = output[:,band] + mask[i,:].T
    return output

def get_git_version():
    try:
        # Get the current commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        
        # Check if there are uncommitted changes
        status = subprocess.check_output(["git", "status", "--porcelain"]).strip().decode('utf-8')
        clean_status = 'Clean' if not status else 'Uncommitted changes'
        
        return commit_hash, clean_status
    except subprocess.CalledProcessError:
        return 'Not a Git repository', 'Unknown'
    
def log_results_to_csv(filename, variable_values, result_values):
    # Get the current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Get the Git version information
    commit_hash, git_status = get_git_version()
    
    # Data to log (you can modify this based on your needs)
    data_to_log = {
        'time': current_time,
        'git_version': commit_hash,
        'git_status': git_status,
        **variable_values,  # Unpack your variable values into the dict
        **result_values     # Unpack your result values into the dict
    }
    
    # Writing to a CSV file, append mode ('a'), with headers only on the first run
    file_exists = False
    try:
        with open(filename, mode='r'):
            file_exists = True
    except FileNotFoundError:
        file_exists = False
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_to_log.keys())
        
        # Write the header only once (if the file does not exist yet)
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow(data_to_log)

def mean_spectral_angle(data1, data2): #returns RMSE spectral angle
    assert data1.shape == data2.shape, "Datacube dimensions not same"
    flat_shape = (-1, data1.shape[-1])  # (num_pixels, num_bands)
    datacube1_flat = data1.reshape(flat_shape)
    datacube2_flat = data2.reshape(flat_shape)
    dot_product = np.sum(datacube1_flat * datacube2_flat, axis=1)
    norm1 = np.linalg.norm(datacube1_flat, axis=1)
    norm2 = np.linalg.norm(datacube2_flat, axis=1)
    epsilon = 1e-9
    norm1 = np.maximum(norm1, epsilon)
    norm2 = np.maximum(norm2, epsilon)
    cosine_angle = dot_product / (norm1 * norm2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    spectral_angle = np.arccos(cosine_angle)
    return np.mean(spectral_angle)

def calculate_psnr(original, reconstructed, max_pixel_value=1.0):
    if original.shape != reconstructed.shape:
        raise ValueError("Input images must have the same dimensions.")
    mse = np.mean((original - reconstructed) ** 2)
    
    if mse == 0:
        return float('inf')  # Infinite PSNR if no difference between images

    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr