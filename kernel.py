import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from scipy import ndimage
from scipy.optimize import minimize
from time import time


def matrix_kernel_estimation_improved(B, I, kernel_size, lambda_reg=0.1, max_iter=200, tol=1e-6):
    """
    Estimate the blur kernel K using constrained least-squares optimization
    with direct minimization of the objective function
    
    Args:
        B: The blurred image (2D numpy array)
        I: The initialization image (2D numpy array)
        kernel_size: The size of the kernel (int)
        lambda_reg: Tikhonov regularization parameter (default: 0.1)
        max_iter: Maximum number of iterations (default: 100)
        tol: Tolerance for convergence (default: 1e-6)
    
    Returns:
        k: The estimated kernel as a 2D array
    """
    # Convert blurred image to vector form
    b = B.flatten()
    I = np.pad(I, kernel_size//2, mode='constant', constant_values=0.0)
    print(f"Input image I shape: {I.shape}")
    print(f"Blurred image B shape: {B.shape}")
    
    # Generate matrix A using view_as_windows
    window_shape = (kernel_size, kernel_size)
    
    try:
        windows = view_as_windows(I, window_shape, step=1)
        print(f"Windows shape: {windows.shape}")
    except ValueError as e:
        print(f"Error creating windows: {e}")
        raise
    
    # Reshape windows to create matrix A
    A = windows.reshape(-1, kernel_size * kernel_size)
    
    # Verify dimensions
    assert A.shape[0] == b.shape[0], \
        f"Matrix A rows ({A.shape[0]}) don't match vector b length ({b.shape[0]})"
    
    print(f"Matrix A dimensions: {A.shape}")
    print(f"Vector b dimensions: {b.shape}")
    
    # Pre-compute constant matrices for efficiency
    ATA = A.T @ A
    ATb = A.T @ b
    
    # Objective function to minimize: ||Ak - b||^2 + lambda||k||^2
    def objective(k):
        return np.sum((A @ k - b)**2) + lambda_reg * np.sum(k**2)
    
    # More efficient objective using pre-computed matrices
    def objective_efficient(k):
        return (k @ ATA @ k - 2 * k @ ATb + b @ b + lambda_reg * (k @ k))
    
    # Gradient of the objective function
    def gradient(k):
        return 2 * (ATA @ k - ATb + lambda_reg * k)
    
    # Constraints: kernel values sum to 1 and are non-negative
    # Define as constraint dictionary for scipy.optimize.minimize
    constraints = [
        {'type': 'eq', 'fun': lambda k: np.sum(k) - 1.0}  # Sum to 1
    ]
    
    # Bounds: all elements must be non-negative
    bounds = [(0, None) for _ in range(kernel_size * kernel_size)]
    
    # Initialize with a better starting point - uniform kernel
    k0 = np.ones(kernel_size * kernel_size) / (kernel_size * kernel_size)
    
    print("Starting optimization...")
    start_time = time()
    
    # Use SLSQP optimization method which handles bounds and constraints well
    result = minimize(
        objective_efficient,
        k0,
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': tol, 'disp': True}
    )
    
    end_time = time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Success: {result.success}, Status: {result.message}")
    print(f"Function evaluations: {result.nfev}")
    
    # Get the optimal kernel
    k_optimal = result.x
    
    # Reshape the kernel to 2D
    k_2d = k_optimal.reshape(kernel_size, kernel_size)
    
    return k_2d


def demonstrate_improved_kernel_estimation():
    """
    Demonstrate the improved kernel estimation with a synthetic example
    """
    # Create a simple image - keeping it small for demonstration
    img_size = (21, 21)
    I = np.zeros(img_size)
    I[5:16, 5:16] = 1.0  # White square on black background
    
    # Create a simple motion blur kernel
    kernel_size = 21
    true_kernel = np.zeros((kernel_size, kernel_size))
    """for i in range(kernel_size):
        true_kernel[i, kernel_size//2] = 1.0 / kernel_size"""
    true_kernel[0:kernel_size//2,0:kernel_size//2] = np.eye(kernel_size//2)
    true_kernel[kernel_size//2+1:, 0:kernel_size//2] = np.eye(kernel_size//2)
    # Normalize the kernel
    true_kernel = true_kernel / np.sum(true_kernel)
    
    # Apply the blur
    B = ndimage.convolve(I, true_kernel, mode='constant', cval=0.0)
    
    # Add a small amount of noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.01, B.shape)
    B_noisy = B + noise
    
    # Print shapes
    print(f"Original image shape: {I.shape}")
    print(f"Blurred image shape: {B_noisy.shape}")
    print(f"Kernel shape: {true_kernel.shape}")
    
    # Estimate the kernel using both methods
    print("\n=== Original Method ===")
    try:
        estimated_kernel_original = matrix_kernel_estimation(
            B_noisy, I, kernel_size, lambda_reg=5.0, beta=1.0, max_iter=100
        )
    except Exception as e:
        print(f"Original method failed: {e}")
        estimated_kernel_original = np.zeros_like(true_kernel)
    
    print("\n=== Improved Method ===")
    estimated_kernel_improved = matrix_kernel_estimation_improved(
        B_noisy, I, kernel_size, lambda_reg=0.1, max_iter=200
    )
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(2, 4, 2)
    plt.imshow(B_noisy, cmap='gray')
    plt.title('Blurred Image')
    
    plt.subplot(2, 4, 3)
    plt.imshow(true_kernel, cmap='gray')
    plt.title('True Kernel')
    
    plt.subplot(2, 4, 4)
    plt.imshow(estimated_kernel_original, cmap='gray')
    plt.title('Original Estimated Kernel')
    
    plt.subplot(2, 4, 5)
    plt.imshow(estimated_kernel_improved, cmap='gray')
    plt.title('Improved Estimated Kernel')
    
    plt.subplot(2, 4, 6)
    plt.plot(true_kernel[kernel_size//2, :], label='True')
    plt.plot(estimated_kernel_original[kernel_size//2, :], label='Original')
    plt.plot(estimated_kernel_improved[kernel_size//2, :], label='Improved')
    plt.legend()
    plt.title('Kernel Cross-section')
    
    # Reconstruct images using both kernels
    recon_original = ndimage.convolve(I, estimated_kernel_original, mode='constant', cval=0.0)
    recon_improved = ndimage.convolve(I, estimated_kernel_improved, mode='constant', cval=0.0)
    
    plt.subplot(2, 4, 7)
    plt.imshow(recon_original, cmap='gray')
    plt.title('Recon with Original Kernel')
    
    plt.subplot(2, 4, 8)
    plt.imshow(recon_improved, cmap='gray')
    plt.title('Recon with Improved Kernel')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate errors
    error_original = np.linalg.norm(estimated_kernel_original - true_kernel) / np.linalg.norm(true_kernel)
    error_improved = np.linalg.norm(estimated_kernel_improved - true_kernel) / np.linalg.norm(true_kernel)
    
    print(f"Original method relative error: {error_original:.4f}")
    print(f"Improved method relative error: {error_improved:.4f}")
    
    return estimated_kernel_improved, true_kernel


# Retain the original function for comparison
def matrix_kernel_estimation(B, I, kernel_size, lambda_reg=5.0, beta=1.0, max_iter=30, tol=1e-6):
    """
    Original kernel estimation function for comparison
    """
    b = B.flatten()
    I = np.pad(I, kernel_size//2, mode='constant', constant_values=0.0)
    print(f"Input image I shape: {I.shape}")
    print(f"Blurred image B shape: {B.shape}")
    
    window_shape = (kernel_size, kernel_size)
    try:
        windows = view_as_windows(I, window_shape, step=1)
        print(f"Windows shape: {windows.shape}")
    except ValueError as e:
        print(f"Error creating windows: {e}")
        raise
    
    A = windows.reshape(-1, kernel_size * kernel_size)
    assert A.shape[0] == b.shape[0]
    
    ATA = A.T @ A
    ATb = A.T @ b
    I_reg = np.eye(kernel_size * kernel_size)
    system_matrix = ATA + lambda_reg**2 * I_reg
    
    k = np.zeros(kernel_size * kernel_size)
    center_idx = (kernel_size**2) // 2
    k[center_idx] = 1.0
    
    print(f"Matrix A dimensions: {A.shape}")
    print(f"Vector b dimensions: {b.shape}")
    print(f"Vector k dimensions: {k.shape}")
    
    start_time = time()
    for i in range(max_iter):
        k_prev = k.copy()
        k = k + beta * (ATb - system_matrix @ k)
        k[k < 0] = 0
        k_sum = np.sum(k)
        if k_sum > 0:
            k = k / k_sum
        
        diff = np.linalg.norm(k - k_prev)
        if diff < tol:
            print(f"Converged at iteration {i+1} with difference {diff:.8f}")
            break
            
    end_time = time()
    print(f"Kernel estimation completed in {end_time - start_time:.2f} seconds")
    print(f"Total iterations: {i+1}")
    
    k_2d = k.reshape(kernel_size, kernel_size)
    return k_2d


if __name__ == "__main__":
    estimated_kernel, true_kernel = demonstrate_improved_kernel_estimation()
    
    # Calculate error between true and estimated kernels
    error = np.linalg.norm(estimated_kernel - true_kernel) / np.linalg.norm(true_kernel)
    print(f"Relative error: {error:.4f}")