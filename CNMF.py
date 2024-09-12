import numpy as np
from VCA_master.VCA import vca
from utilities import Cost, normalize

def CNMF(HSI_data: np.array, MSI_data: np.array, spatial_transform: np.array, spectral_response: np.array, endmembers=40, loops=(200,5), tol=0.01) -> np.array:
    """Upscales HSI image to MSI resolution using coupled non-negative matrix factorisation
        
    Args:
        HSI_data (np.array): Low spatial/high spectral resolution HSI data (x,y,b)
        MSI_data (np.array): High spatial/low spectral resolution MSI data (x,y,b)

    Returns:
        tuple[np.array,np.array]: Upscaled HSI data
    """
    precision = np.float32
    h_bands, m_bands = HSI_data.shape[2], MSI_data.shape[2]
    HSI_data = np.clip(HSI_data,1E-15,np.max(HSI_data)) #Should this be necessary?
    MSI_data = np.clip(MSI_data,1E-15,np.max(MSI_data)) #Should this be necessary?
    CheckMat(HSI_data, "HSI", zero=True)
    CheckMat(MSI_data, "MSI", zero=True)
    h_flat, m_flat = HSI_data.reshape(h_bands,-1), MSI_data.reshape(m_bands, -1)

    w = np.zeros(shape=(h_bands,endmembers),dtype=precision)
    h = np.ones(shape=(endmembers,m_flat.shape[1]),dtype=precision)/endmembers

    w_m = np.zeros(shape=(m_bands, endmembers))
    h_h = np.ones(shape=(endmembers,h_flat.shape[1]),dtype=precision)/endmembers

    Ae, _, _ = vca(h_flat, endmembers, verbose=True, snr_input=10)

    #STEP 1
    w[:,:] = Ae[:,:]
    h_h = PixelSumToOne(h_h*np.matmul(w.transpose(),h_flat)/np.matmul(w.transpose(),np.matmul(w,h_h)))
    done_i, done_o, count_i, count_o, = False, False, 0, 0
    i_in, i_out = loops[0], loops[1]
    last_i =  1E-15
    #Optimise?
    while done_o != True:
        CheckMat(np.matmul(w,h_h),"w*h_h", zero=True)
        CheckMat(np.matmul(w,h), "wxh", zero=True)
        #STEP 2a
        w_m = np.matmul(spectral_response, w)
        h = PixelSumToOne(h*np.matmul(w_m.transpose(),m_flat)/np.matmul(w_m.transpose(),np.matmul(w_m,h))) #Loop?
        done_i = False
        count_i = 0
        last_i = 1E-15
        while done_i != True:
            #STEP 2b
            w_m = w_m*np.matmul(m_flat,h.transpose())/np.matmul(w_m,np.matmul(h,h.transpose()))
            h = PixelSumToOne(h*np.matmul(w_m.transpose(),m_flat)/np.matmul(w_m.transpose(),np.matmul(w_m,h)))#Loop?
            cost = Cost(m_flat, np.matmul(w_m,h))
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
        h_h = PixelSumToOne(np.matmul(h,spatial_transform))
        w = w*np.matmul(h_flat,h_h.transpose())/np.matmul(w,np.matmul(h_h,h_h.transpose()))#Loop?
        while done_i != True:
            #STEP 2b           
            h_h = PixelSumToOne(h_h*np.matmul(w.transpose(),h_flat)/np.matmul(w.transpose(),np.matmul(w,h_h)))
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
    out_flat = np.matmul(w,h)
    out = out_flat.reshape(MSI_data.shape[0],MSI_data.shape[1],h_bands)
    return out

def CheckMat(data, name, zero = False):
    assert not np.any(np.isinf(data)), f"Matrix {name} has infinite values"
    assert not np.any(np.isnan(data)), f"Matrix {name} has NaN values"
    if zero:
        assert not np.any(data == 0), f"Matrix {name} has Zero values"

def PixelSumToOne(data: np.array) -> np.array:
    assert len(data.shape) == 2, "Array incorrectly dimensioned"
    pixel_sums = data.sum(axis=0)
    return data/pixel_sums