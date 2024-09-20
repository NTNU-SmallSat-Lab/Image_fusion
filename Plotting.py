import numpy as np
import utilities as util
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from io import BytesIO
from PIL import Image

def save_spec_error(data1, data2, ax):
    levels = np.loadtxt("RGB_mask.txt")*100
    spec_error = get_spectral_error(data1, data2)
    ax2 = ax.twinx()
    ax.plot(spec_error, linestyle='-', label='Spectral error sum', color='orange')
    start_nm, end_nm = util.band_to_wavelength(4)[0].astype(int), util.band_to_wavelength(116)[1].astype(int)
    ax2.plot(np.linspace(4,116,end_nm-start_nm),levels[start_nm:end_nm, 0], linestyle='dotted', color='red', label='Red Channel')
    ax2.plot(np.linspace(4,116,end_nm-start_nm),levels[start_nm:end_nm, 1], linestyle='dotted', color='green', label='Green Channel')
    ax2.plot(np.linspace(4,116,end_nm-start_nm),levels[start_nm:end_nm, 2], linestyle='dotted', color='blue', label='Blue Channel')
    ax2.vlines([103], ymin=0, ymax=spec_error.max())
    #plt.vlines(x=[72, 43, 15], ymin=0, ymax=spec_error.max(), colors=['r', 'g', 'b'])
    ax.set_xlabel('bands')
    ax.set_ylabel('Error [%]')
    ax2.set_ylabel('Quantum efficiency [%]')
    plt.grid(True)
    plt.legend(loc='upper left')

def save_endmembers(endmembers, abundances, shape, path): #Make this better
    endmember_count = endmembers.shape[1]
    if int(endmember_count/5) == 0:
        array = np.zeros(shape=(2*shape[0],endmember_count*shape[1],4))
    else:
        array = np.zeros(shape=(2*int(endmember_count/5)*shape[0],5*shape[1],4))
    abundance_array = np.zeros_like(array)
    for i in range(endmember_count):
        plt.figure(figsize=(shape[0]/100,shape[1]/100), dpi=100, facecolor='white', layout='compressed')
        plt.plot(endmembers[:,i])
        plt.axis('off')
        plt.vlines([103],ymin=0, ymax=endmembers[:,i].max())
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
    util.save_RGB((Data*255).astype(np.uint8), name)
    return

def save_final_image(Original: np.array, downscaled: np.array, Upscaled: np.array, spectral_response_matrix: np.array, save_path: str):
    fig = plt.figure(figsize=(10,15))
    gs = fig.add_gridspec(3,2)
    input = Normalize(np.matmul(Original, spectral_response_matrix.T))
    downscaled = Normalize(np.matmul(downscaled, spectral_response_matrix.T))
    output = Normalize(np.matmul(Upscaled, spectral_response_matrix.T))

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2,:])
    
    ax1.imshow(input, interpolation='none')
    ax1.set_title("Original")
    ax1.axis('off')

    ax2.imshow(downscaled, interpolation='none')
    ax2.set_title("Downscaled")
    ax2.axis('off')

    ax3.imshow(output, interpolation='none')
    ax3.set_title("Upscaled")
    ax3.axis('off')

    get_error(Original, Upscaled, ax4)

    save_spec_error(Original, Upscaled, ax5)

    plt.tight_layout()
    plt.savefig(f"{save_path}Output", bbox_inches='tight', dpi=300)
    plt.close(fig)
    

def get_error(data1, data2, ax):
    error_percent = np.mean(100 * np.abs((data1 - data2) / (data1+1E-9)), axis=2)
    cax = ax.imshow(error_percent, interpolation='none', cmap='gray', norm=LogNorm())
    cbar = plt.colorbar(cax, ax=ax)
    min_val = np.min(error_percent[error_percent > 0])  # Smallest non-zero value
    max_val = np.max(error_percent)
    ticks = np.logspace(np.log10(min_val), np.log10(max_val), num=5)  # Logarithmic spacing of ticks
    cbar.set_ticks(ticks)
    cbar.ax.minorticks_off()
    cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
    ax.set_title("Error [%]")
    ax.axis('off')

def get_spectral_error(data1, data2):
     assert not np.any(np.isinf(data1)), "Data1 has infinite values"
     assert not np.any(np.isnan(data1)), "Data1 has nan values"
     assert not np.any(np.isinf(data2)), "Data2 has infinite values"
     assert not np.any(np.isnan(data2)), "Data2 has nan values"
     data1, data2 = np.clip(data1, a_min=1E-6, a_max=1.0), np.clip(data2, a_min=1E-6, a_max=1.0)
     with np.errstate(divide='ignore', invalid='ignore'):
        error = np.abs((data1 - data2) / data1)
     print(f"Max error: {error.max()}")
     summed_error = np.mean((100*error), axis=(0,1))
     return summed_error

def Normalize(data, min=0.0, max=1.0):
    data_min, data_max = data.min(), data.max()
    data = (data-data_min)*(max-min)/(data_max-data_min)
    return data