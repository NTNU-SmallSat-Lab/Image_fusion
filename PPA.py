import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import loader as ld
import utilities as util
import Plotting as plot
import os
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean

class simple_PPA:
    def __init__(self, data, n, delta=0.15) -> None:
        self.w = np.ones(shape=(data.shape[0], n)) / np.sqrt(data.shape[0])
        self.h = np.ones(shape=(n, data.shape[1])) / n
        self.endmembers = n
        self.weights = np.ones(data.shape[1])
        self.delta = delta

    def single_member_update(self, data, i):
        err = (data - self.w@self.h).T
        rem = (data - self.w[:,i][:,np.newaxis]).T #remainder without EM contribution'

        elo_sum = np.sum(np.multiply(self.weights*self.h[i], err.T), axis=1) #calculates the total error for each band
        elo = elo_sum@rem.T
        denom_2_sum = self.weights*self.h[i]**2
        denom = np.sum(denom_2_sum)

        norm = np.array([k@k for k in rem])
        a = np.ones_like(norm, dtype=np.float64)
        #print(f"denom: ({denom.shape},{denom.sum()})\nnorm: ({norm.shape},{norm.sum()})\nelo: ({elo.shape},{elo.sum()})")
        total_change = - (denom*norm)/2*a**2 + elo.T*a.T 
        energies = -total_change

        sor_eng = np.argsort(energies)
        #print(energies.min())
        j = 1
        # Define a threshold for spectral similarity
        threshold = 0.15

        remE = [k for k in range(self.endmembers)]
        remE.remove(i)

        # Initial pixel spectrum
        pixel_spectrum = data[:, sor_eng[j]]

        # Check if the pixel spectrum is too similar to any existing spectra
        in_group = np.any([euclidean(pixel_spectrum, self.w[:, j]) < threshold for j in remE])

        while in_group:
            if j == data.shape[1] - 1:  # Break if at the last pixel
                break
            j += 1
            pixel_spectrum = data[:, sor_eng[j]]  # Get the next ranked spectrum

            # Check similarity again with the updated spectrum
            in_group = np.any([euclidean(pixel_spectrum, self.w[:, j]) < threshold for j in remE])

        # If the selected spectrum has sufficient energy, assign it
        if energies[sor_eng[j]] < 0:
            self.w[:, i] = data[:, sor_eng[j]]

    
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

def get_PPA(data, EM, delta=0.15):
    flat = data.reshape(data.shape[0]*data.shape[1],data.shape[2]).T

    sppa = simple_PPA(flat, EM, delta)
    sppa.train(flat)
    return sppa.w, sppa.h

if __name__ == "__main__":
    data_string, name = util.Get_path()
    EM = 3

    x_start, x_end, y_start, y_end = 100, 300, 0, 200
    pix_coords = [x_start,x_end,y_start,y_end]
    size = (x_end-x_start, y_end-y_start)

    arr = ld.load_l1b_cube(data_string, coords=pix_coords)
    arr = util.remove_darkest(arr)
    arr = plot.Normalize(arr)
    flat = arr.reshape(arr.shape[0]*arr.shape[1],arr.shape[2]).T

    sppa = simple_PPA(flat, EM, delta=0.05)

    sppa.train(flat)

    save_path = f"outputs\\PPA{name}_{x_start}-{x_end}x_{y_start}-{y_end}y\\"
    if not os.path.exists(save_path):
            os.mkdir(save_path)

    rgb_mask = np.loadtxt("RGB_mask.txt")
    spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)
    upscaled = np.matmul(sppa.w,sppa.h).T.reshape(arr.shape[0], arr.shape[1], arr.shape[2])

    plot.save_endmembers_few(sppa.w, sppa.h, size, save_path)
    plot.save_final_image(arr, np.ones_like(arr), upscaled, spectral_response_matrix, save_path)
