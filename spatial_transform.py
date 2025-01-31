import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def align_and_overlay(hsi_path, rgb_path, output_path): #TODO rewrite this to crop and recalculate transform to only include active area
    hsi_img = cv2.imread(hsi_path, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    rgb_img = cv2.GaussianBlur(rgb_img, (15, 15), 0)

    cv2.imwrite(f"{output_path}rgb.png", rgb_img)
    cv2.imwrite(f"{output_path}hsi.png", hsi_img)
    
    # Initialize SIFT detector - parameters are almost definitely sub-optimal
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=20, sigma=2)

    # Detect keypoints and descriptors
    kp_hsi, des_hsi = sift.detectAndCompute(hsi_img, None)
    kp_rgb, des_rgb = sift.detectAndCompute(rgb_img, None)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_hsi, des_rgb, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    img_matches = cv2.drawMatches(hsi_img, kp_hsi, rgb_img, kp_rgb, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    
    plt.savefig(f"{output_path}matches.png")
    plt.close()

    # Extract the matching keypoints
    src_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_hsi[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5,confidence=0.995, maxIters=5000)
    
    print(M)
    
    inliers = mask.ravel() == 1
    outliers = mask.ravel() == 0
    # Checking how many matches are outliers
    #print(f"{np.sum(inliers)} inliers and {np.sum(outliers)} outliers")

    # Transform
    aligned_rgb = cv2.warpPerspective(rgb_img, M, (hsi_img.shape[1], hsi_img.shape[0]))

    # Save the aligned image
    cv2.imwrite(f"{output_path}aligned_rgb.png", aligned_rgb)

    # blend images
    overlay_img = cv2.addWeighted(hsi_img, 0.5, aligned_rgb, 0.5, 0)
    
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
    n_pixel = 24
    Source = np.full(shape=(hsi_dim[0], hsi_dim[1], n_pixel, 2), dtype=np.int16, fill_value=np.nan)
    for i in range(rgb_dim[0]):
        for j in range(rgb_dim[1]):
            Destcoord = transformation@np.array([i, j, 1])
            Dest[i,j,:] = np.array([int(Destcoord[0]),int(Destcoord[1])])
            if 0 <= int(Destcoord[0]) < hsi_dim[0] and 0 <= int(Destcoord[1]) < hsi_dim[1]:
                k = 0
                #print(int(Destcoord[0]),int(Destcoord[1]))
                while Source[int(Destcoord[0]),int(Destcoord[1]),k,0] != -1:
                    k += 1
                    assert k < (n_pixel), "Array size inadequate"
                Source[int(Destcoord[0]),int(Destcoord[1]),k,:] = np.array([i,j])
    
    return Source

def generate_spatial_subset(area_of_interest, full_transform):
    subtransform = full_transform[area_of_interest[0]:area_of_interest[1],area_of_interest[2]:area_of_interest[3],:,:]
    rgb_x = np.max(subtransform[:,:,:,0])-np.min(subtransform[:,:,:,0])
    rgb_y = np.max(subtransform[:,:,:,1])-np.min(subtransform[:,:,:,1])
    
    hsi_x = area_of_interest[1] - area_of_interest[0]
    hsi_y = area_of_interest[3] - area_of_interest[2]    
    
    transform_array = np.zeros(shape=(rgb_x*rgb_y,hsi_x*hsi_y))
    
    for i in range(subtransform.shape[0]):
        for j in range(subtransform.shape[1]):
            pass

    
if __name__ == "__main__":
    hsi_path = "input/HSI.png"
    rgb_path = "input/RGB.png"
    output_path = "output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    transform = align_and_overlay(hsi_path=hsi_path, rgb_path=rgb_path, output_path=output_path)
    Source = generate_full_spatial((4784,1092),(3840,2748), transform)