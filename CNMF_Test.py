import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from VCA_master.VCA import vca
import loader as ld
from saver import save_HSI_as_RGB
import utilities as util

data_string = ".\\data\\APEX_OSD_Package_1.0\\APEX_osd_V1_calibr_cube"

precision = np.float32

rgb = {
    "R": 44,
    "G": 18,
    "B": 5
}

arr = ld.load_envi(precision=precision, path=data_string, x_start=300, x_end=400, y_start=300, y_end=400)
arr = util.normalize(arr)
size = arr.shape
arr_flattened = arr.reshape(size[2],size[0]*size[1])

save_HSI_as_RGB(arr, name="Original.png", rgb=rgb)

rgb_representation = arr[:,:,[rgb["R"],rgb["G"],rgb["B"]]]
rgb_flattened = rgb_representation.reshape(3,size[0]*size[1])

sigma = 1
downsampling_factor = 2
assert (size[0]%downsampling_factor == 0) and (size[1]%downsampling_factor == 0), "Resolution must be whole multiple of downsample factor"
downscaled_size = (int(size[0]/downsampling_factor), int(size[1]/downsampling_factor), size[2])

blurred = gaussian_filter(arr, sigma=(sigma, sigma, 0), mode='reflect').astype(precision)
lowres_downsampled = np.zeros(shape=downscaled_size)
save_HSI_as_RGB(blurred, name="Downsampled.png", rgb=rgb)
lowres_downsampled[:,:,:] = blurred[::downsampling_factor,::downsampling_factor,:]
lowres_downsampled_flattened = lowres_downsampled.reshape(downscaled_size[2],downscaled_size[1]*downscaled_size[0])
save_HSI_as_RGB(lowres_downsampled, name="Downscaled.png", rgb=rgb)
endmember_count = 40
bands = lowres_downsampled_flattened.shape[0]
pixels_h = lowres_downsampled_flattened.shape[1]
pixels_m = rgb_flattened.shape[1]

spatial_transform_matrix = np.zeros(shape=(pixels_m,pixels_h),dtype=precision)
for i in range(downscaled_size[0]):
    spatial_transform_matrix[downsampling_factor*i:downsampling_factor*(i+1)-1,i] = 1 #combine pixels 2-to-1 NEEDS UPDATING FOR DIFFERENT DOWNSAMPLING VALUES
spectral_spread = 1
spectral_response_matrix = np.zeros(shape=(3,bands), dtype=precision)
spectral_response_matrix[0,rgb["R"]-spectral_spread:rgb["R"]+spectral_spread+1] = 1
spectral_response_matrix[1,rgb["G"]-spectral_spread:rgb["G"]+spectral_spread+1] = 1
spectral_response_matrix[2,rgb["B"]-spectral_spread:rgb["B"]+spectral_spread+1] = 1 #sum up a few bands around each RGB component since no calibration data
spectral_response_matrix = spectral_response_matrix/spectral_response_matrix.sum(axis=1, keepdims=True)

w = np.zeros(shape=(bands,endmember_count),dtype=precision)
h = np.ones(shape=(endmember_count,pixels_m),dtype=precision)/endmember_count #endmember and abundance matrices for fused data

w_m = np.zeros(shape=(3,endmember_count)) #reduced band endmember spectrum matrix
h_h = np.ones(shape=(endmember_count,pixels_h),dtype=precision)/endmember_count #reduced resolution abundance matrix

Ae, _, _ = vca(lowres_downsampled_flattened, endmember_count, verbose = True) #CHECK VCA FUNCTION/WRITE OWN
#STEP 1
w[:,:] = Ae[:,:]
h_h = h_h*np.matmul(w.transpose(),lowres_downsampled_flattened)/np.matmul(w.transpose(),np.matmul(w,h_h))
done_i, done_o, count_i, count_o, = False, False, 0, 0
i_in, i_out = 50, 80
last_i, last_o =  1E-15, 1E-15
tol_i, tol_o = 0.000000002, 0
assert w.shape == (bands,endmember_count), "W has dimensions {w.shape}, should have ({bands},{endmember_count})"
assert w_m.shape == (3,endmember_count), "W_m has dimensions {w_m.shape}, should have (3,{endmember_count})"
assert h.shape == (endmember_count,pixels_m), "H has dimensions {h.shape}, should have ({endmember_count},{original_resolution})"
assert h_h.shape == (endmember_count,pixels_h), "H_h has dimensions {h_h.shape}, should have ({endmember_count},{downsampled_resolution})"
while done_o != True:
    #STEP 2
    w_m = np.matmul(spectral_response_matrix,w)
    h = h*np.matmul(w_m.transpose(),rgb_flattened)/np.matmul(w_m.transpose(),np.matmul(w_m,h))
    last_i = 1E-12
    count_i = 0
    done_i = False
    while done_i != True:
        w_m = w_m*np.matmul(rgb_flattened,h.transpose())/np.matmul(w_m,np.matmul(h,h.transpose()))
        h = h*np.matmul(w_m.transpose(),rgb_flattened)/np.matmul(w_m.transpose(),np.matmul(w_m,h))
        cost = util.Cost(rgb_flattened, np.matmul(w_m,h))
        count_i += 1
        if (last_i-cost)/last_i<tol_i:
            done_i = True
        else:
            last_i = cost
        if count_i == i_in:
            print("Counted out")
            done_i = True
    #STEP 3
    done_i = False
    count_i = 0
    last_i = 1E-12
    h_h = np.ones(shape=(endmember_count,pixels_h),dtype=precision)/endmember_count
    while done_i != True:
        w = w*np.matmul(lowres_downsampled_flattened,h_h.transpose())/np.matmul(w,np.matmul(h_h,h_h.transpose()))
        h_h = h_h*np.matmul(w.transpose(),lowres_downsampled_flattened)/np.matmul(w.transpose(),np.matmul(w,h_h))
        cost = util.Cost(lowres_downsampled_flattened, np.matmul(w,h_h))
        count_i += 1
        if (last_i-cost)/last_i<tol_i:
            done_i = True
        else:
            last_i = cost
        if count_i == i_in:
            print("Counted out")
            done_i = True
    count_o += 1
    if count_o == i_out:
        done_o = True
result = np.matmul(w,h).reshape(size[0],size[1],size[2])
#error = np.abs(result-arr_flattened)

#save_HSI_as_RGB(error,"Error.png", rgb=rgb)
save_HSI_as_RGB(result, "Output.png", rgb=rgb)