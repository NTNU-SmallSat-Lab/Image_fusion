import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import utilities as util
def align_and_overlay(hsi_img, rgb_img, output_path): #TODO rewrite this to crop and recalculate transform to only include active area
    rgb_img = cv2.GaussianBlur(rgb_img, (15, 15), 0)
    #rgb_img = rgb_img[::3,::3]
    rgb_img = cv2.flip(rgb_img, 1)

    cv2.imwrite(f"{output_path}rgb.png", rgb_img)
    cv2.imwrite(f"{output_path}hsi.png", hsi_img)
    
    # Initialize SIFT detector - parameters are almost definitely sub-optimal
    sift = cv2.SIFT_create(nfeatures=50000, contrastThreshold=0.015, edgeThreshold=40, sigma=2)

    # Detect keypoints and descriptors
    kp_hsi, des_hsi = sift.detectAndCompute(hsi_img, None)
    kp_rgb, des_rgb = sift.detectAndCompute(rgb_img, None)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_hsi, des_rgb, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    img_matches = cv2.drawMatches(hsi_img, kp_hsi, rgb_img, kp_rgb, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    
    plt.savefig(f"{output_path}matches.png")
    plt.close()

    # Extract the matching keypoints
    src_pts = np.float32([kp_hsi[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5,confidence=0.995, maxIters=5000)
    
    print(M)
    
    inliers = mask.ravel() == 1
    outliers = mask.ravel() == 0
    # Checking how many matches are outliers
    #print(f"{np.sum(inliers)} inliers and {np.sum(outliers)} outliers")

    # Transform
    aligned_hsi = cv2.warpPerspective(hsi_img, M, (rgb_img.shape[1], rgb_img.shape[0]))

    # Save the aligned image
    cv2.imwrite(f"{output_path}aligned_hsi.png", aligned_hsi)

    # blend images
    overlay_img = cv2.addWeighted(rgb_img, 0.5, aligned_hsi, 0.5, 0)
    
    cv2.imwrite(f"{output_path}overlay.png", overlay_img)
    return M
    
def generate_full_spatial(hsi_dim,rgb_dim, transformation):
    """Heavy function that generates source map for each HSI pixel. The idea being that the heavy lifting
        will only be performed once, then the source map can be accessed quickly for each patch during fusion.

    Args:
        hsi_dim (tuple): hsi_dimensions
        rgb_dim (tuple): rgb_dimensions
        transformation (np.array): homogeneous transformation matrix (3,3)
    """
    #Dest maps where each RGB pixel is mapped on HSI, for example Dest[250, 250] = [2045, 145]
    #Indicates that RGB pixel (250,250) is spatially equivelent to HSI pixel (2045, 145)
    #For the moment these are being brute forced into specific pixels using Int() THIS SHOULD BE FIXED
    Dest = np.zeros(shape=(rgb_dim[0], rgb_dim[1], 2), dtype=np.int16)
    n_pixel = 25
    Source = np.ones(shape=(hsi_dim[0], hsi_dim[1], n_pixel, 2), dtype=np.int16)*-1
    for i in range(rgb_dim[0]):
        if i%10 == 0:
            print(f"Row {i}")
        for j in range(rgb_dim[1]):
            Destcoord = transformation@np.array([i, j, 1])
            Dest[i,j,:] = np.array([int(Destcoord[0]),int(Destcoord[1])])
            if 0 <= int(Destcoord[0]) < hsi_dim[0] and 0 <= int(Destcoord[1]) < hsi_dim[1]:
                k = 0
                #print(int(Destcoord[0]),int(Destcoord[1]))
                while Source[int(Destcoord[0]), int(Destcoord[1]), k, 0] != -1:
                    k += 1
                    assert k < n_pixel, "Array size inadequate"
                Source[int(Destcoord[0]),int(Destcoord[1]),k,:] = np.array([i,j])
    
    return Source

def generate_spatial_subset(area_of_interest, full_transform):
    #need to implement method to correct values from each pixel, at the moment each pixel contributes equally
    subtransform = full_transform[area_of_interest[0]:area_of_interest[1],area_of_interest[2]:area_of_interest[3],:,:]
    maskx = subtransform[:,:,:,0] != -1
    masky = subtransform[:,:,:,1] != -1
    print(f"np.max(x) = {np.max(subtransform[:,:,:,0])}, np.min(x) = {np.min(subtransform[:,:,:,0][maskx])}")
    print(f"np.max(y) = {np.max(subtransform[:,:,:,1])}, np.min(y) = {np.min(subtransform[:,:,:,1][masky])}")
    rgb_x_min, rgb_x_max = np.min(subtransform[:,:,:,0][maskx]), np.max(subtransform[:,:,:,0])
    rgb_y_min, rgb_y_max = np.min(subtransform[:,:,:,1][masky]), np.max(subtransform[:,:,:,1])
    rgb_x = rgb_x_max-rgb_x_min
    rgb_y = rgb_y_max-rgb_y_min
    
    hsi_x = area_of_interest[1] - area_of_interest[0]
    hsi_y = area_of_interest[3] - area_of_interest[2]    
    
    transform_array = np.zeros(shape=((rgb_x+1)*(rgb_y+1),hsi_x*hsi_y))
    
    for i in range(subtransform.shape[0]):
        for j in range(subtransform.shape[1]):
            k = 0
            while subtransform[i,j,k,0] != -1:
                try:
                    transform_array[
                        subtransform[i, j, k, 0] - rgb_x_min + (subtransform[i, j, k, 1] - rgb_y_min) * rgb_x,
                        i + j * hsi_x
                    ] = 1  # TODO: Implement weighting

                except IndexError as e:
                    raise IndexError(
                        f"Out-of-bounds access in transform_array: "
                        f"Index ({subtransform[i, j, k, 0] - rgb_x_min}, {subtransform[i, j, k, 1] - rgb_y_min}) \n"
                        f"Transformed index: {subtransform[i, j, k, 0] - rgb_x_min + (subtransform[i, j, k, 1] - rgb_y_min) * rgb_x, i + j * hsi_x}\n"
                        f"From: {subtransform[i, j, k, 0] - rgb_x_min} + {(subtransform[i, j, k, 1] - rgb_y_min)}*{rgb_x}\n"
                        f"with shape {transform_array.shape}. Check index calculations."
                    ) from e

                except Exception as e:
                    raise RuntimeError(f"Unexpected error in array indexing: {e}") from e

                    
                k += 1
    
    normalization_weights = np.sum(transform_array, axis=0)
    transform_array = transform_array/normalization_weights
    print(transform_array)
    return transform_array

def get_pixels(pixel_bounds, transformation, rgb_dim):
    hsi_x_min, hsi_x_max = pixel_bounds[0], pixel_bounds[1]
    hsi_y_min, hsi_y_max = pixel_bounds[2], pixel_bounds[3]
    
    hsi_x, hsi_y = hsi_x_max-hsi_x_min, hsi_y_max-hsi_y_min
    
    origin = transformation@np.array([hsi_x_min, hsi_y_min, 1.0])
    extent = transformation@np.array([hsi_x_max, hsi_y_max, 1.0])
    
    rgb_x_min, rgb_x_max = int(np.floor(origin[0])), int(np.ceil(extent[0]))
    rgb_y_min, rgb_y_max = int(np.floor(origin[1])), int(np.ceil(extent[1]))
    
    rgb_x, rgb_y = rgb_x_max-rgb_x_min, rgb_y_max-rgb_y_min
    
    hsi_pixels = hsi_x*hsi_y
    rgb_pixels = rgb_x*rgb_y
    
    transform = np.zeros(shape=(rgb_pixels, hsi_pixels))
    
    for i in range(hsi_pixels):
        x = i%hsi_x
        y = i//hsi_x
        area_origin = transformation@np.array([x+hsi_x_min, y+hsi_y_min, 1.0])
        area_extent = transformation@np.array([x+1+hsi_x_min, y+1+hsi_y_min, 1.0])
        x_min, x_max = int(np.floor(area_origin[0]))-rgb_x_min, int(np.ceil(area_extent[0]))-rgb_x_min
        y_min, y_max = int(np.floor(area_origin[1]))-rgb_y_min, int(np.ceil(area_extent[1]))-rgb_y_min
        for j in range(x_max-x_min):
            for k in range(y_max-y_min):
                transform[j+x_min+(k+y_min)*rgb_x,i] = 1.0
                print(f"assigning link between RGB ({j+x_min},{k+y_min}) and HSI ({x},{y})")
    
    normalization_weights = np.sum(transform, axis=0)
    transform = transform/normalization_weights
    return transform
    
def full_transform(rgb_img, HSI_datacube):
    rgb_mask = np.loadtxt("RGB_mask.txt")
    spectral_response_matrix = util.map_mask_to_bands(rgb_mask[0:700,:],112)
    hsi_img = np.matmul(HSI_datacube,spectral_response_matrix.T)
    hsitorgb = align_and_overlay(hsi_img, rgb_img)
    rgbtohsi = cv2.invert(hsitorgb)
    origin = rgbtohsi@np.array([0, 0, 1.0])
    extent = rgbtohsi@np.array([rgb_img.shape[0], rgb_img.shape[1], 1.0])
    x_min = np.clip(origin[0], a_min=0.0, a_max=hsi_img.shape[0]) #this still assumes entire bounded square is defined
    x_max = np.clip(extent[0], a_min=0.0, a_max=hsi_img.shape[0])
    y_min = np.clip(origin[1], a_min=0.0, a_max=hsi_img.shape[1])
    y_max = np.clip(extent[1], a_min=0.0, a_max=hsi_img.shape[1])
    hsi_img = hsi_img[x_min:x_max, y_min:y_max]
    transform = align_and_overlay(hsi_img, rgb_img) #lazy solution
    active_area = np.array([x_min, x_max, y_min, y_max])
    return active_area, transform
    
if __name__ == "__main__":
    #hsi_path, _ = util.Get_path("HSI image")
    hsi_path = "slice.png"
    #rgb_path, _ = util.Get_path("RGB image")
    rgb_path = "data\\bluenile_2025_01_25T08_23_14.png"
    hsi_img = cv2.normalize(cv2.imread(hsi_path, cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    rgb_img = cv2.normalize(cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    output_path = "output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    transform = align_and_overlay(hsi_img, rgb_img, output_path=output_path)
    #transform_subset = get_pixels(np.array([500, 503, 500, 503]), transform, (0,0))
    #print(transform_subset.shape)
    