import numpy as np
from utilities import save_RGB
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def save_spec_error(spec_error, path):
    plt.figure(figsize=(8,4))
    plt.plot(spec_error, linestyle='-', label='Spectral error sum')
    plt.vlines(x=[72, 43, 15], ymin=0, ymax=spec_error.max(), colors=['r', 'g', 'b'])
    plt.title('Spectral error')
    plt.xlabel('bands')
    plt.ylabel('Error [%]')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{path}\\Spectral_error.png")

def save_endmembers(endmembers, abundances, shape, path):
    endmember_count = endmembers.shape[1]
    if int(endmember_count/5) == 0:
        array = np.zeros(shape=(2*shape[0],endmember_count*shape[1],4))
    else:
        array = np.zeros(shape=(2*int(endmember_count/5)*shape[0],5*shape[1],4))
    abundance_array = np.zeros_like(array)
    for i in range(endmember_count):
        plt.figure(figsize=(shape[0]/100,shape[1]/100), dpi=100, facecolor='white', layout='compressed')
        plt.plot(endmembers[:,i])
        plt.xlabel("Bands", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        buf = BytesIO()
        plt.savefig(buf)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        image_array = np.asarray(img)
        abundance_map = np.stack([abundances[i,:].T.reshape(shape[0],shape[1],-1)[:,:,0]] * 4, axis=-1)
        abundance_map = (abundance_map*255)
        abundance_map[:,:,3] = np.ones_like(abundance_map[:,:,3])*255
        array[int(i/5)*shape[0]*2:int(i/5)*shape[0]*2+shape[0],(i%5)*shape[1]:(i%5+1)*shape[1],:] = image_array
        abundance_array[int(i/5)*shape[0]*2+shape[0]:int(i/5+1)*shape[0]*2,(i%5)*shape[1]:(i%5+1)*shape[1],:] = abundance_map
    abundance_array[:,:,0:3]=abundance_array[:,:,0:3]*255/abundance_array[:,:,0:3].max()
    array = array+abundance_array
    img = Image.fromarray(array.astype(np.uint8))
    img.save(f"{path}\\endmembers.png")

def save_HSI_as_RGB(Data, name):
    assert len(Data.shape) == 3, "Data array is not 3 dimensional"
    save_RGB((Data*255).astype(np.uint8), name)
    return