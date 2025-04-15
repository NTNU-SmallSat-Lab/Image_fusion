import numpy as np
from PPA import simple_PPA

def CPPA(HSI_data: np.array, 
         MSI_data: np.array, 
         spatial_transform: np.array, 
         spectral_response: np.array,
         endmember_list,
         delta=0.15,
         endmembers=3, 
         loops=(10,5),
         tol=1e-2,
         target=1.0) -> list[np.array, np.array, np.array]:
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

    h_flat, m_flat = HSI_data, MSI_data
    
    #print(h_flat.shape, m_flat.shape)
    
    h_ppa = simple_PPA(data=h_flat, delta=delta, n=endmembers, target = target, endmembers=endmember_list)
    m_ppa = simple_PPA(data=m_flat, delta=delta, n=endmembers, target = target, endmembers=endmember_list)

    w, h = main_loop(h_ppa, 
                     m_ppa, 
                     h_flat, 
                     m_flat, 
                     spectral_response, 
                     spatial_transform, 
                     loops[1], 
                     tol)
    out = np.matmul(w,h)
    return out, w, h


def PPA_HSI_step(PPA_obj: simple_PPA, data):
    PPA_obj.all_endmembers_update(data)

def PPA_to_RGB(PPA_obj: simple_PPA, transform):
    return transform@PPA_obj.w

def RGB_NNLS(PPA_obj: simple_PPA, data):
    PPA_obj.abundances_update(data)

def Fuse_RGB_to_HSI(PPA_obj: simple_PPA, transform):
    #print(PPA_obj.h.shape)
    #print(transform.shape)
    return PPA_obj.h@transform

def main_loop(PPA_obj_h: simple_PPA, 
              PPA_obj_m: simple_PPA, 
              data_h: np.array, 
              data_m: np.array, 
              spectral_transform: np.array, 
              spatial_transform: np.array, 
              loops: int,
              tol: float):
    for i in range(loops):
        #print(f"Starting loop {i}")
        PPA_HSI_step(PPA_obj_h, data_h)
        PPA_obj_m.w = PPA_to_RGB(PPA_obj_h, spectral_transform)
        RGB_NNLS(PPA_obj_m, data_m)
        PPA_obj_h.h = Fuse_RGB_to_HSI(PPA_obj_m, spatial_transform)
    return PPA_obj_h.w, PPA_obj_m.h