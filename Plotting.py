import numpy as np
import utilities as util
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

def save_spec_error(data1, data2, ax):
    levels = np.loadtxt("RGB_mask.txt")*100
    spec_error = util.calculate_psnr(data1,data2,axis=(0,1))
    ax2 = ax.twinx()
    start_nm, end_nm = util.band_to_wavelength(4)[0].astype(int), util.band_to_wavelength(116)[1].astype(int)
    ax.plot(np.linspace(start_nm,end_nm,data1.shape[2]),spec_error, linestyle='-', label='Spectral error sum', color='orange')
    ax2.plot(np.linspace(start_nm,end_nm,end_nm-start_nm),levels[start_nm:end_nm, 0], linestyle='dotted', color='red')
    ax2.plot(np.linspace(start_nm,end_nm,end_nm-start_nm),levels[start_nm:end_nm, 1], linestyle='dotted', color='green')
    ax2.plot(np.linspace(start_nm,end_nm,end_nm-start_nm),levels[start_nm:end_nm, 2], linestyle='dotted', color='blue')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Peak SNR [dB]')
    ax2.set_ylabel('Quantum efficiency [%]')
    plt.grid(True)

def save_endmembers_many(endmembers, abundances, shape, save_path):
    assert endmembers.shape[1] > 10, "Too few endmembers use save_endmembers_few()"
    count = endmembers.shape[1]
    order = []
    def scientific_notation(x, pos):
        return f'{x:.3f}'
    for i in range(count):
        max_abundance = 0
        selected = -1
        for j in range(count):
            abundance = np.sum(abundances[j,:])
            if (abundance > max_abundance) and j not in order:
                max_abundance = abundance
                selected = j
        if selected != -1:
            order.append(selected)
    rows = int((count-1)/5)+1
    fig = plt.figure(figsize=(15,6*rows))
    gs = fig.add_gridspec(2*rows,5)
    ax_endmembers = []
    ax_abundances = []
    for i in range(count):
        row, column = 2*int(i/5), i%5
        ax_endmembers.append(fig.add_subplot(gs[row,column]))
        ax_abundances.append(fig.add_subplot(gs[row+1,column]))
        abundance_map = abundances[order[i]].T.reshape(shape[0],shape[1],-1)
        min_val = np.min(abundance_map[abundance_map > 0.01])  # Smallest non-zero value
        max_val = np.max(abundance_map)
        ax_endmembers[i].plot(endmembers[:,order[i]])
        cax = ax_abundances[i].imshow(abundance_map, interpolation='none', norm=LogNorm(vmin=0.01, vmax=1), cmap='viridis')
        cbar = plt.colorbar(cax, ax=ax_abundances[i])
        cbar.ax.minorticks_off()
        ticks = np.logspace(np.log10(min_val), np.log10(max_val), num=3)
        cbar.set_ticks(ticks)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation))
        ax_abundances[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_path}Endmembers", bbox_inches='tight', dpi=300)
    plt.close(fig)

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

def save_endmembers_few(endmembers, abundances, shape, save_path):
    assert endmembers.shape[1] <= 10, "Too many endmembers use save_endmembers_many()"
    count = endmembers.shape[1]
    if count < 5:
        columns = count
    else:
        columns = 5
    rows = int((count-1)/5)+1
    fig = plt.figure(figsize=(20,5+5*rows))
    gs = fig.add_gridspec(1+rows,columns)
    order = []
    for i in range(count):
        max_abundance = 0
        selected = -1
        for j in range(count):
            abundance = np.sum(abundances[j,:])
            if (abundance > max_abundance) and j not in order:
                max_abundance = abundance
                selected = j
        if selected != -1:
            order.append(selected)

    def scientific_notation(x, pos):
        return f'{x:.3f}'
    
    start_nm, end_nm = util.band_to_wavelength(4)[0].astype(int), util.band_to_wavelength(116)[1].astype(int)
    
    ax = []
    ax.append(fig.add_subplot(gs[0,:]))
    lines = [('-',1), ('dashed', 1), ('dotted',2)]
    for i in range(count):
        line = lines[i%3]
        ax[0].plot(np.linspace(start_nm,end_nm,endmembers.shape[0]),endmembers[:,order[i]],label=f"Spectrum {i}", linestyle=line[0], lw=line[1])
        ax[0].set_xlabel('Wavelength [nm]')
    ax[0].legend()
    
    for i in range(count):
        ax.append(fig.add_subplot(gs[1+int(i/5),i%5]))
        abundance_map = abundances[order[i]].T.reshape(shape[0],shape[1],-1)
        #min_val = np.min(abundance_map[abundance_map > 0.001])  # Smallest non-zero value
        #max_val = np.max(abundance_map)
        #cax = ax[i+1].imshow(abundance_map, interpolation='none', norm=LogNorm(vmin=0.001, vmax=1), cmap='viridis')
        cax = ax[i+1].imshow(abundance_map, interpolation='none', cmap='viridis')
        cbar = plt.colorbar(cax, ax=ax[i+1])
        #ticks = np.logspace(np.log10(min_val), np.log10(max_val), num=5)
        #cbar.set_ticks(ticks)
        cbar.ax.minorticks_off()
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation))
        ax[i+1].set_title(f"Spectrum {i} abundances")
        ax[i+1].axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_path}Endmembers", bbox_inches='tight', dpi=300)
    plt.close(fig)

    

def get_error(data1, data2, ax):
    error = util.mean_spectral_angle(data1, data2, map=True)
    #min_val = np.min(error)  # Smallest non-zero value
    #max_val = np.max(error)
    #cax = ax.imshow(error, interpolation='none', cmap='viridis', norm=LogNorm())
    cax = ax.imshow(error, interpolation='none', cmap='viridis')
    cbar = plt.colorbar(cax, ax=ax)
    #ticks = np.logspace(np.log10(min_val), np.log10(max_val), num=5)  # Logarithmic spacing of ticks
    #cbar.set_ticks(ticks)
    cbar.ax.minorticks_off()
    #cbar.set_ticklabels([f'{tick:.3f}' for tick in ticks])
    ax.set_title("Spectral angle [rad]")
    ax.axis('off')

def Normalize(data, min=0.0, max=1.0):
    data_min, data_max = data.min(), data.max()
    data = (data-data_min)*(max-min)/(data_max-data_min)+min
    return data

