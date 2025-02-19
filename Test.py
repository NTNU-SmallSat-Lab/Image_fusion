from spatial_transform import get_distribution
import numpy as np
import matplotlib.pyplot as plt

def create_gaussian_array(size=100, mean=0, std=1):
    """
    Create a 100x100 numpy array with a normalized 2D Gaussian distribution.
    
    Parameters:
    size (int): The size of the square array (default: 100)
    mean (float): Mean of the Gaussian distribution (default: 0)
    std (float): Standard deviation of the Gaussian distribution (default: 1)
    
    Returns:
    np.array: 2D normalized Gaussian array
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # 2D Gaussian function
    Z = np.exp(-((X**2 + Y**2) / (2 * std**2)))
    
    # Normalize the Gaussian
    Z /= np.sum(Z)
    
    return Z

# Create the Gaussian array
gaussian_array = create_gaussian_array()

print(np.sum(get_distribution((2.1, 2.4),gaussian_array)))
