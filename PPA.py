import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import loader as ld
import utilities as util
import Plotting as plot
import os

class simple_PPA:
    def __init__(self, data, n) -> None:
        self.w = np.ones(shape=(data.shape[0],n))/np.sqrt(data.shape[0])
        self.h = np.ones(shape=(n,data.shape[1]))/n
        self.weights = np.ones_like(data)
        self.endmembers = n
        self.delta = 0.15

    def single_member_update(self, data, index):
        """err = data - np.matmul(self.w,self.h) TODO FIX THIS
        new_w = self.w
        new_w[:,index] = 0
        err_no_i = data - np.matmul(new_w,self.h)
        dot_products = np.dot(err_no_i, data.T)
        sorted_pixel_indices = np.argsort(np.diag(dot_products))[::-1]
        second_best = sorted_pixel_indices[1]
        self.w[:,index] = data[:,second_best].T
        diff, second, first, second_i, first_i = 0, 1.1E12, 1E12, -1, -1
        for i in range(data.shape[1]):
            new_w[:,index] = data[:,i].T
            diff = np.linalg.norm(err - np.matmul(new_w,self.h), ord='fro')**2
            if diff < first:
                second = first
                first = diff
                second_i = first_i
                first_i = i
            elif diff < second:
                second = diff
                second_i = i
        if second == -1:
            return
        else:
            self.w[:,index] = data[:,second_i].T"""
    
    def all_endmembers_update(self, data):
        for i in range(self.endmembers):
            self.single_member_update(data, i)
    
    def abundances_update(self, data):
        w_e = np.ones(shape=(self.w.shape[0]+1,self.w.shape[1]))
        w_e[:-1,:] = self.delta*self.w
        data_e = np.ones(shape=(data.shape[0]+1,data.shape[1]))
        data_e[:-1,:] = self.delta*data

        S = np.array([opt.nnls(w_e, i, maxiter=100)[0] for i in data_e.T], dtype=np.float64).transpose()
        self.h = (S + self.h)/2
    
    def obj(self, data):
        return np.linalg.norm(data-np.matmul(self.w,self.h), ord='fro')**2
    
    def train(self, data):
        obj = self.obj(data)
        iter_since_best = 0
        best_obj = 2*obj
        print(obj)
        while iter_since_best < 10:
            self.all_endmembers_update(data)
            self.abundances_update(data)
            obj = self.obj(data)
            if self.obj(data) < best_obj:
                best_obj = self.obj(data)
                if obj < best_obj - 100:
                    print("Iter_since_best: 0")
                    iter_since_best = 0
            elif obj > best_obj - 100:
                iter_since_best += 1
                print(f"Iter_since_best: {iter_since_best}")
            print(obj)


data_string, name = util.Get_path()
EM = 5

x_start, x_end, y_start, y_end = 0, 100, 0, 100
pix_coords = [x_start,x_end,y_start,y_end]
size = (x_end-x_start, y_end-y_start)

arr = ld.load_l1b_cube(data_string, coords=pix_coords)
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
