import numpy as np
from PPA import simple_PPA

def CPPA(HSI_data: np.array, 
         MSI_data: np.array, 
         spatial_transform: np.array, 
         spectral_response: np.array,
         endmember_list = None,
         delta=0.15,
         endmembers=3, 
         loops=(10,5),
         tol=1e-2,
         target=1.0) -> list[np.array, np.array, np.array]:
    
    h_ppa = simple_PPA(data=HSI_data, delta=delta, n=endmembers, target = target, endmembers=endmember_list)
    m_ppa = simple_PPA(data=MSI_data, delta=delta, n=endmembers, target = target, endmembers=endmember_list)

    w, h = main_loop(h_ppa, 
                     m_ppa, 
                     HSI_data, 
                     MSI_data, 
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
        PPA_HSI_step(PPA_obj_h, data_h)
        PPA_obj_m.w = PPA_to_RGB(PPA_obj_h, spectral_transform)
        RGB_NNLS(PPA_obj_m, data_m)
        PPA_obj_h.h = Fuse_RGB_to_HSI(PPA_obj_m, spatial_transform)
    return PPA_obj_h.w, PPA_obj_m.h