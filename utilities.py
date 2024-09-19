import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path
from PIL import Image, ImageOps
from math import sqrt
from scipy.ndimage import gaussian_filter

def normalize(image):
        """
        Normalize an image array to the range [0, 255].
        """
        img_min = np.max([np.min(image), 1e-15])
        img_max = np.max(image)
        # Avoid division by zero if all pixels are the same
        output = image/(img_max-img_min)
        return output

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

def Wavelength_to_band(lmda, sat_obj):
        return np.argmin(abs(sat_obj.spectral_coefficients - lmda))

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

def get_error(data1, data2):
    error_percent = np.mean(100*np.abs((data1-data2)/data1), axis=2)
    error_log = np.clip(np.log(error_percent*100), a_min=0, a_max=None)
    perspective_bar = np.linspace(start=0, stop=np.log(10000), num=data1.shape[0])
    error_log[:,-10:] = np.tile(perspective_bar, (10,1)).T
    error_log = (error_log/error_log.max())
    return error_log

def get_spectral_error(data1, data2):
     error = np.abs(data1-data2)
     summed_error = np.mean((100*error/data1), axis=(0,1))
     return summed_error