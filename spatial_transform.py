import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def align_and_overlay(hsi_path, rgb_path, output_path):
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
    
def generate_full_spatial(hsi_dim, rgb_dim, transformation):
    for i in range(rgb_dim[0]):
        for j in range(rgb_dim[1]):
            hsi_coords = np.array[i, j, 1]
    
if __name__ == "__main__":
    hsi_path = "input/HSI.png"
    rgb_path = "input/RGB.png"
    output_path = "output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    align_and_overlay(hsi_path=hsi_path, rgb_path=rgb_path, output_path=output_path)