import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import loader as ld
import utilities as util
import Plotting as plot
import os

def remove_darkest(data):
    data_proc = data
    for i in range(data.shape[2]):
        data_proc -= np.min(data[:,:,i])
    return data_proc

def spectral_angle(spectrum1, spectrum2):
    """Compute the spectral angle (in radians) between two spectra."""
    # Calculate dot product of the two spectra
    dot_product = np.dot(spectrum1, spectrum2)
    
    # Calculate the norms of the spectra
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)
    
    # Compute the cosine of the spectral angle
    cos_theta = dot_product / (norm1 * norm2)
    
    # To avoid numerical issues, clip the cosine value to the range [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Compute the angle in radians
    angle = np.arccos(cos_theta)
    
    return angle

class simple_PPA:
    def __init__(self, data, n) -> None:
        self.w = np.ones(shape=(data.shape[0],n))/np.sqrt(data.shape[0])
        self.h = np.ones(shape=(n,data.shape[1]))/n
        self.weights = np.ones_like(data)
        self.endmembers = n
        self.delta = 0.3
        self.angle_tolerance = np.radians(15)

    def single_member_update(self, data, i):
        err = data - np.matmul(self.w,self.h)
        contrib = np.matmul(self.w[:,i][:,np.newaxis], self.h[i][np.newaxis,:])
        rem = err - contrib

        elo_sum = np.sum(np.multiply(self.weights*self.h[i].T, err), axis=1) #calculates the total abundance*error of the chosen endmember over entire image
        elo = elo_sum@rem

        denom_2_sum = self.weights*contrib**2
        denom = np.sum(denom_2_sum)

        norm = np.array([k@k for k in rem.T])
        a = np.ones_like(norm, dtype=np.float64)
        total_change = - (denom*norm)/2*a**2 + elo*a.T 
        energies = -total_change

        sor_eng = np.argsort(energies)
        j = 2
        k = j
        remE = [k for k in range(self.endmembers)]
        remE.remove(i)
        # Get the spectrum for the selected pixel in the ranking
        pixel_spectrum = data[:, sor_eng[j]]

        # Check if the pixel_spectrum is already in the remaining endmembers
        in_group = np.any([spectral_angle(pixel_spectrum, self.w[:,k]) < self.angle_tolerance for k in remE])

        # Loop to find a spectrum not already in `self.w[remE]`
        while in_group:
            if k == (data.shape[1] - 1) or energies[sor_eng[k]] > 0:
                return
            k += 1
            pixel_spectrum = data[:, sor_eng[k]]  # Get the next ranked spectrum
            in_group = np.any([spectral_angle(pixel_spectrum, self.w[:,k]) < self.angle_tolerance for k in remE])
        
        if j > 1:
            self.w[:,i] = data[:,sor_eng[j]]
        else:
            if energies[sor_eng[j]]<0:
                self.w[:,i] = data[:,sor_eng[j]]

    
    def all_endmembers_update(self, data):
        for i in range(self.endmembers):
            self.single_member_update(data, i)
    
    def abundances_update(self, data):
        w_e = np.ones(shape=(self.w.shape[0]+1,self.w.shape[1]))
        w_e[:-1,:] = self.delta*self.w
        data_e = np.ones(shape=(data.shape[0]+1,data.shape[1]))
        data_e[:-1,:] = self.delta*data

        S = np.array([opt.nnls(w_e, i, maxiter=5000)[0] for i in data_e.T], dtype=np.float64).transpose()
        self.h = (S + self.h)/2
    
    def obj(self, data):
        return np.sum((data-self.w@self.h)**2)
    
    def train(self, data, tol=1e-2):
        obj = self.obj(data)
        old_obj = 2*obj
        dobj = (old_obj-obj)/(old_obj+obj)
        print(obj)
        while np.abs(dobj) > tol:
            self.all_endmembers_update(data)
            self.abundances_update(data)
            old_obj = obj
            obj = self.obj(data)
            dobj = (old_obj-obj)/(old_obj+obj)
            print(obj)


data_string, name = util.Get_path()
EM = 3

x_start, x_end, y_start, y_end = 0, 80, 0, 80
pix_coords = [x_start,x_end,y_start,y_end]
size = (x_end-x_start, y_end-y_start)

arr = ld.load_l1b_cube(data_string, coords=pix_coords)
arr = remove_darkest(arr)
arr = plot.Normalize(arr)
flat = arr.reshape(arr.shape[0]*arr.shape[1],arr.shape[2]).T

sppa = simple_PPA(flat, EM)

sppa.train(flat)

save_path = f"outputs\\PPA{name}_{x_start}-{x_end}x_{y_start}-{y_end}y\\"
if not os.path.exists(save_path):
        os.mkdir(save_path)

rgb_mask = np.loadtxt("RGB_mask.txt")
spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)
upscaled = np.matmul(sppa.w,sppa.h).T.reshape(arr.shape[0], arr.shape[1], arr.shape[2])

plot.save_endmembers_few(sppa.w, sppa.h, size, save_path)
plot.save_final_image(arr, np.ones_like(arr), upscaled, spectral_response_matrix, save_path)
