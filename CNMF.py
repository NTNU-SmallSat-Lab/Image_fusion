import numpy as np
from VCA_master.VCA import vca
from utilities import Cost
import loader as ld
from scipy.optimize import minimize

def CNMF(HSI_data: np.array, 
         MSI_data: np.array, 
         spatial_transform: np.array, 
         spectral_response: np.array, 
         VCA_init: np.array,
         delta=0.9,
         endmembers=40, 
         loops=(10,5),
         tol=0.00005) -> list[np.array, np.array, np.array]:
    """Performs coupled Nonnegative matrix factorisation to upscale HSI data using spatial data from MSI

    Args:
        HSI_data (np.array): low spatial/high spectral resolution datacube shape=(x_h, y_h, b_h)
        MSI_data (np.array): high spatial/low spectral resolution datacube shape=(x_m, y_m, b_m)
        spatial_transform (np.array): flattened Spatial transform from HSI to MSI shape=(x_m*y_m,x_h*y_h)
        spectral_response (np.array): spectral transform from HSI to MSI shape=(b_m, b_h)
        VCA_init (np.array): Endmember matrix initialization shape=(b_h, endmembers)
        endmembers (int, optional): number of endmembers. Defaults to 40.
        loops (tuple, optional): inner and outer loops. Defaults to (200,5).
        tol (float, optional): inner loop cost tolerance. Defaults to 0.00005.

    Returns:
        list[np.array, np.array, np.array]: upscaled datacube, endmember spectra, abundances
    """

    precision = np.float64
    h_bands, m_bands = HSI_data.shape[2], MSI_data.shape[2]

    HSI_data = np.clip(HSI_data,1E-15,np.max(HSI_data)) #Should this be necessary?
    MSI_data = np.clip(MSI_data,1E-15,np.max(MSI_data)) #Should this be necessary?
    CheckMat(HSI_data, "HSI", zero=True)
    CheckMat(MSI_data, "MSI", zero=True)

    #Flatten arrays, add sum-to-one requirement
    h_flat, m_flat = np.ones(shape=(HSI_data.shape[0]*HSI_data.shape[1],h_bands+1)).T, np.ones(shape=(MSI_data.shape[0]*MSI_data.shape[1],m_bands+1)).T
    h_flat[:-1,:], m_flat[:-1,:] = delta*HSI_data.reshape(HSI_data.shape[0]*HSI_data.shape[1],h_bands).T, delta*MSI_data.reshape(-1,m_bands).T

    w = np.ones(shape=(h_bands+1,endmembers),dtype=precision)

    h = np.ones(shape=(endmembers,m_flat.shape[1]),dtype=precision)/endmembers
    
    w_m = np.ones(shape=(spectral_response.shape[0]+1,w.shape[1]))
    w_m[:-1,:] = delta*np.matmul(spectral_response, w[:-1,:])
    h_h = np.matmul(h,spatial_transform)

    #STEP 1a
    w[:-1,:] = delta*VCA_init
    CheckMat(np.matmul(w,h_h),"w*h_h", zero=True)
    h_h = h_h*np.matmul(w.transpose(),h_flat)/np.matmul(w.transpose(),np.matmul(w,h_h))
    done_i, done_o, count_i, count_o, = False, False, 0, 0
    i_in, i_out = loops[0], loops[1]
    last_i =  1E-15
    while done_i != True:
        #STEP 1b
        w[-1,:] = np.ones_like(w[-1,:])
        h_h = h_h*np.matmul(w.transpose(),h_flat)/np.matmul(w.transpose(),np.matmul(w,h_h))
        w = w*np.matmul(h_flat,h_h.transpose())/np.matmul(w,np.matmul(h_h,h_h.transpose()))
        cost = Cost(h_flat[:-1,:], np.matmul(w[:-1,:],h_h))
        count_i += 1
        if abs((last_i-cost)/last_i) < tol:
            done_i = True
        else:
            last_i = cost
        if count_i == i_in:
            done_i = True
    while done_o != True:
        print(f"Run {count_o}")
        #STEP 2a
        w_m[:-1,:] = np.matmul(spectral_response, w[:-1,:])
        w_m[-1,:] = np.ones_like(w_m[-1,:])
        h = h*np.matmul(w_m.transpose(),m_flat)/np.matmul(w_m.transpose(),np.matmul(w_m,h)) #Loop?
        done_i = False
        count_i = 0
        last_i = 1E-15
        while done_i != True:
            #STEP 2b
            w_m = w_m*np.matmul(m_flat,h.transpose())/np.matmul(w_m,np.matmul(h,h.transpose()))
            w_m[-1,:] = np.ones_like(w_m[-1,:])
            h = h*np.matmul(w_m.transpose(),m_flat)/np.matmul(w_m.transpose(),np.matmul(w_m,h))#Loop?
            cost = Cost(m_flat[:-1,:], np.matmul(w_m[:-1,:],h))
            count_i += 1
            if abs((last_i-cost)/last_i) < tol:
                done_i = True
            else:
                last_i = cost
            if count_i == i_in:
                done_i = True
        done_i = False
        count_i = 0
        last_i = 1E-15
        #Step 3a
        w[-1,:] = np.ones_like(w[-1,:])
        h_h = np.matmul(h,spatial_transform)
        w = w*np.matmul(h_flat,h_h.transpose())/np.matmul(w,np.matmul(h_h,h_h.transpose()))#Loop?
        while done_i != True:
            #STEP 3b
            w[-1,:] = np.ones_like(w[-1,:]) 
            h_h = h_h*np.matmul(w.transpose(),h_flat)/np.matmul(w.transpose(),np.matmul(w,h_h))
            w = w*np.matmul(h_flat,h_h.transpose())/np.matmul(w,np.matmul(h_h,h_h.transpose()))
            cost = Cost(h_flat, np.matmul(w,h_h))
            count_i += 1
            if abs((last_i-cost)/last_i) < tol:
                done_i = True
            else:
                last_i = cost
            if count_i == i_in:
                done_i = True
        count_o += 1
        if count_o == i_out:
            done_o = True
        """print(f"Mean h per pixel sum: {np.mean(np.sum(h, axis=0))}")
        print(f"Max h per pixel sum: {np.max(np.sum(h, axis=0))}")
        print(f"Min h per pixel sum: {np.min(np.sum(h, axis=0))}")"""
    out_flat = np.matmul(w[:-1,:],h)
    out = out_flat.T.reshape(MSI_data.shape[0], MSI_data.shape[1], h_bands)
    return out, w[:-1,:], h

def CheckMat(data: np.array, name: str, zero = False):
    """Simple check to ensure matrix well defined

    Args:
        data (np.array): Matrix to be checked
        name (string): String to identify matrix in output
        zero (bool, optional): Whether to check for zeros. Defaults to False.
    """
    assert not np.any(np.isinf(data)), f"Matrix {name} has infinite values"
    assert not np.any(np.isnan(data)), f"Matrix {name} has NaN values"
    if zero:
        assert not np.any(data == 0), f"Matrix {name} has Zero values"

def PixelSumToOne(data: np.array) -> np.array:
    """Modify matrix so sum of each column vector is 1

    Args:
        data (np.array): matrix shape.len=2

    Returns:
        np.array: matrix with vectors summed to one
    """
    assert len(data.shape) == 2, "Array incorrectly dimensioned"
    pixel_sums = np.sum(data, axis=0)
    new_data = data/pixel_sums
    return new_data

def Get_VCA(string: str, endmembers: int, coords=[0,0,0,0], bands=[0,0]):
    """Retrieve endmembers of an l1b datacube using vertex component analysis

    Args:
        string (str): path to l1b cube
        endmembers (int): number of endmembers
        coords (list, optional): Area to retrieve endmembers from (x_start, x_end, y_start, y_end). Defaults to entire datacube.

    Returns:
        np.array: returns spectral signature matrix shape=(bands,endmembers)
    """
    data = ld.load_l1b_cube(string)
    if coords != [0,0,0,0]:
        data=data[coords[0]:coords[1],coords[2]:coords[3],:]
    if bands != [0,0]:
        data = data[:,:,bands[0]:bands[1]]
    h_flat = data.reshape(data.shape[0]*data.shape[1],data.shape[2]).T
    Ae, _, _ = vca(h_flat, endmembers, verbose=True)
    return Ae