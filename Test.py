import numpy as np
import utilities as util
import loader as ld
import matplotlib.pyplot as plt

def range(data1, data2):
    max1 = np.max(data1, axis=(0,1))
    max2 = np.max(data2, axis=(0,1))
    
    min1 = np.min(data1, axis=(0,1))
    min2 = np.min(data2, axis=(0,1))

    range1 = np.max(data1, axis=(0,1))-np.min(data1, axis=(0,1))
    range2 = np.max(data2, axis=(0,1))-np.min(data2, axis=(0,1))

    plt.figure(figsize=(10,10))
    plt.plot(max1, color='r', linestyle='dotted', lw=0.4, label="Original max")
    plt.plot(min1, color='r', linestyle='dashed', lw=0.4, label="Original min")
    plt.plot(max2, color='b', linestyle='dotted', lw=0.4, label="Upscaled max")
    plt.plot(min2, color='b', linestyle='dashed', lw=0.4, label="Upscaled min")
    plt.plot(range1, color='r', label="Original Range")
    plt.plot(range2, color='b', label="Upscaled Range")
    plt.legend(loc='upper left')
    plt.show()