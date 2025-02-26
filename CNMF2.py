import numpy as np
from VCA_master.VCA import vca
from utilities import Cost

class CNMF:
    def __init__(self,
                HSI_data: np.array, 
                MSI_data: np.array, 
                spatial_transform: np.array, 
                spectral_response: np.array, 
                delta=0.9,
                endmembers=40, 
                loops=(10,5),
                tol=0.00005):
        self.w_init = Get_VCA(HSI_data, endmembers)
        self.HSI_data = HSI_data
        self.MSI_data = MSI_data
        self.spatial_transform = spatial_transform
        self.spectral_response = spectral_response
        self.delta = delta
        self.endmembers = endmembers
        self.loops = loops
        self.tol = tol
        self.extend_arrays()
        self.w = np.ones(shape=(self.HSI_data.shape[0], self.endmembers))
        self.h = np.ones(shape=(self.endmembers,self.MSI_data.shape[1]))/self.endmembers
        self.w_m = np.ones(shape=(self.MSI_data.shape[0], self.endmembers))
        self.w_m[:-1,:] = delta*(self.spectral_response@self.w[:-1,:])
        self.h_h = np.ones(shape=(self.endmembers, self.HSI_data.shape[1]))/self.endmembers
        self.final = None
        self.main_loop()
    
    def extend_arrays(self) -> None: #extra row of ones for sum-to-one constraint
        self.MSI_data = np.vstack((self.delta * self.MSI_data, np.ones((1, self.MSI_data.shape[1]))))
        self.HSI_data = np.vstack((self.delta * self.HSI_data, np.ones((1, self.HSI_data.shape[1]))))
        
    def Step_1A(self) -> None:
        self.w[:-1,:] = self.delta*self.w_init
        self.h_h = self.h_h*(self.w.T@self.HSI_data)/(self.w.T@(self.w@self.h_h))
    
    def Step_1B_3B(self) -> float:
        #STEP 1b and 3b (identical)
        self.w[-1,:] = np.ones_like(self.w[-1,:])
        self.h_h = self.h_h*(self.w.T@self.HSI_data)/(self.w.T@(self.w@self.h_h))
        self.w = self.w*(self.HSI_data@self.h_h.T)/(self.w@(self.h_h@self.h_h.T))
        return Cost(self.HSI_data[:-1,:], (self.w[:-1,:]@self.h_h))
        
    def Step_2A(self) -> None:
        self.w_m[:-1,:] = self.spectral_response@self.w[:-1,:] #delta?
        self.w_m[-1,:] = np.ones_like(self.w_m[-1,:])
        self.h = self.h*(self.w_m.T@self.MSI_data)/(self.w_m.T@(self.w_m@self.h))
        
    def Step_2B(self) -> float:
        self.w_m = self.w_m*(self.MSI_data@self.h.T)/(self.w_m@(self.h@self.h.T))
        self.w_m[-1,:] = np.ones_like(self.w_m[-1,:]) #delta?
        self.h = self.h*(self.w_m.T@self.MSI_data)/(self.w_m.T@(self.w_m@self.h))
        return Cost(self.MSI_data[:-1,:], (self.w_m[:-1,:]@self.h))
    
    def Step_3A(self) -> None:
        self.w[-1,:] = np.ones_like(self.w[-1,:])
        self.h_h = self.h@self.spatial_transform
        self.w = self.w*(self.HSI_data@self.h_h.T)/(self.w@(self.h_h@self.h_h.T))
    
    def main_loop(self):
        self.Step_1A()
        done = False
        last_cost = 1e-15
        count = 0
        while not done and count < self.loops[1]:
            cost = self.Step_1B_3B()
            count += 1
            if abs((last_cost-cost)/last_cost) < self.tol:
                done = True
            last_cost = cost
        done = False
        last_cost = 1e-15
        count = 0
        for _ in range(self.loops[1]):
            self.Step_2A()
            while not done and count < self.loops[0]:
                cost = self.Step_2B()
                count += 1
                if abs((last_cost-cost)/last_cost) < self.tol:
                    done = True
                last_cost = cost
        done = False
        last_cost = 1e-15
        count = 0
        for _ in range(self.loops[1]):
            self.Step_3A()
            while not done and count < self.loops[0]:
                cost = self.Step_1B_3B()
                count += 1
                if abs((last_cost-cost)/last_cost) < self.tol:
                    done = True
                last_cost = cost
        self.final = self.w[:-1,:]@self.h

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

def Get_VCA(h_flat, endmembers: int):
    Ae, _, _ = vca(h_flat, endmembers, verbose=True)
    return Ae
