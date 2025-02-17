import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import utilities as util
def align_and_overlay(hsi_img, rgb_img, output_path):
    rgb_img = cv2.GaussianBlur(rgb_img, (15, 15), 0)

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

    M, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5,confidence=0.995, maxIters=5000)
    N, mask2 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransacReprojThreshold=5,confidence=0.995, maxIters=5000)
    
    inliers = mask1.ravel() == 1
    outliers = mask1.ravel() == 0
    # Checking how many matches are outliers
    #print(f"{np.sum(inliers)} inliers and {np.sum(outliers)} outliers")

    # Transform
    aligned_hsi = cv2.warpPerspective(hsi_img, M, (rgb_img.shape[1], rgb_img.shape[0]))
    aligned_rgb = cv2.warpPerspective(rgb_img, N, (hsi_img.shape[1], hsi_img.shape[0]))

    # Save the aligned image
    cv2.imwrite(f"{output_path}aligned_hsi.png", aligned_hsi)
    cv2.imwrite(f"{output_path}aligned_rgb.png", aligned_rgb)

    # blend images
    overlay_img = cv2.addWeighted(hsi_img, 0.5, aligned_rgb, 0.5, 0)
    
    cv2.imwrite(f"{output_path}overlay.png", overlay_img)
    return M, N

def get_pixels(pixel_bounds, rgb_bounds, transform_r2h, transform_h2r): #THIS WHOLE FUNCTION IS A MESS AND SHOULD PROBABLY BE REWRITTEN
    """Finds spatial transform between HSI and RGB section.

    Args:
        pixel_bounds (np.array): point coords that define HSI active area
        rgb_bounds (np.array): point coords that define entire RGB image
        transform_r2h (np.array): homogenous transform from RGB->HSI
        transform_h2r (np.array): homogenous transform from HSI->RGB

    Returns:
        tuple (np.array, np.array, np.array, np.array): (spatial transform, rgb bounds [xmin, xmax, ymin, ymax], hsi overlap mask, rgb overlap mask)
    """
    
    h_mask = find_overlap(pixel_bounds, rgb_bounds, transform_r2h)
    r_mask = find_overlap(rgb_bounds, pixel_bounds, transform_h2r)
    
    rgb_min_x, rgb_max_x, rgb_min_y, rgb_max_y = find_edges(r_mask)
    
    h_pixels = np.sum(h_mask)
    r_pixels = np.sum(r_mask)
    
    spatial_transform = np.zeros(shape=(h_pixels, r_pixels))
    skipped = 0
    for i in range(rgb_min_x,rgb_max_x):
        for j in range(rgb_min_y,rgb_max_y):
            if r_mask[i,j] == 0:
                skipped += 1
                break
            
            hsi = (transform_r2h@np.array([i,j,1]))[:2]
            hsi = np.uint32(np.round(hsi))
            assert h_mask[hsi[0], hsi[1]] == 1, f"Spatial mapping disagrees with mask"
            spatial_transform[hsi[0]+hsi[1]*h_mask.shape[0],(i-rgb_min_x)+(j-rgb_min_y)*r_mask.shape[0]-skipped] = 1
    
    normalization_weights = np.sum(spatial_transform, axis=0)
    spatial_transform = spatial_transform/normalization_weights
    
    return spatial_transform, np.array([rgb_min_x, rgb_max_x, rgb_min_y, rgb_max_y]), h_mask, r_mask
    
def full_transform(rgb_img, hsi_img):
    """Finds homogenous projective transform between RGB and HSI, crops HSI to minimum size then recalculates transform.

    Args:
        rgb_img (np.array): RGB image (grayscale)
        hsi_img (np.array): HSI image (grayscale)
    Returns:
        tuple: (HSI_limits,RGB_limits, HSI->RGB transform, RGB->HSI transform)
    """
    transform_h2r, transform_r2h = align_and_overlay(hsi_img, rgb_img,"output/") #find both transforms for full images
    hsi_points = np.array([0, hsi_img.shape[1]-1, 0, hsi_img.shape[0]-1])
    rgb_points = np.array([0, rgb_img.shape[1]-1, 0, rgb_img.shape[0]-1])
    util.plot_transforms(hsi_points,rgb_points,transform_r2h)
    #print(f"hsi_points: {hsi_points}\nrgb_points: {rgb_points}")
    h_mask = find_overlap(hsi_points,rgb_points,transform_r2h) #find hsi mask of overlapped area
    hsi_limits = find_edges(h_mask) #find x/y limits of overlapped area
    hsi_img = hsi_img[hsi_limits[0]:hsi_limits[1],hsi_limits[2]:hsi_limits[3]].copy() #crop rest
    hsi_limits = [0,hsi_img.shape[1]-1,0,hsi_img.shape[0]-1] #redefine limits
    transform_h2r, transform_r2h = align_and_overlay(hsi_img, rgb_img,"output/") #find both transforms for cropped HSI
    #print(f"rgb_points: {rgb_points}\nhsi_points: {hsi_limits}")
    r_mask = find_overlap(rgb_points,hsi_limits,transform_h2r) #find RGB mask of overlapped area
    rgb_limits = find_edges(r_mask) #find x/y limits of overlapped area
    util.plot_transforms(rgb_points,hsi_limits,transform_h2r)
    plt.imshow(r_mask)
    plt.show()
    os._exit(0)
    return hsi_limits, rgb_limits, transform_h2r, transform_r2h

def find_overlap(A_points: np.ndarray, B_points: np.ndarray, transform: np.ndarray) -> np.ndarray: #should probably be extended to not assume origin = (0,0)
    """Returns mask of area of A that is covered by B

    Args:
        A_dim (tuple): (height, width) of A
        B_dim (tuple): (height, width) of B
        transform (np.array): homogenous transform from B to A coordinate system

    Returns:
        np.array: (n,2) where n is the number of points defining overlap shape, each point defined by (x,y) in A coordinates
    """
    B_corners = np.array([
        [B_points[0], B_points[2], 1.0], #x0, y0
        [B_points[1], B_points[2], 1.0], #x1, y0
        [B_points[1], B_points[3], 1.0], #x1, y1
        [B_points[0], B_points[3], 1.0]  #x0, y1
    ]).astype(np.float32)

    B_corners_warped_homogeneous = (transform @ B_corners.T).T
    B_corners_warped = (B_corners_warped_homogeneous[:, :2] / B_corners_warped_homogeneous[:, 2:3])
    
    B_corners_warped[:, 0] -= A_points[0]  # Adjust x
    B_corners_warped[:, 1] -= A_points[2]  # Adjust y
    
    poly_corners = []
    
    for i in range(B_corners_warped.shape[0]):
        start_coord = B_corners_warped[i]
        if i + 1 == B_corners_warped.shape[0]:
            end_coord = B_corners_warped[0]
        else:
            end_coord = B_corners_warped[i+1]
        segment_intersections = find_intersect(A_points, start_coord, end_coord)
        poly_corners.extend(segment_intersections)
    poly_corners = np.array(poly_corners, dtype=np.int32)
    poly_corners = order_points_clockwise(poly_corners)
    
    poly_corners = poly_corners.reshape((-1, 1, 2))
    mask = np.zeros((A_points[3]-A_points[2]+1, A_points[1]-A_points[0]+1), dtype=np.uint8)
    print(f"\n===OVERLAP===\npoly corners: {poly_corners}\n mask size: {mask.shape}\nWarped corners = {B_corners_warped}")
    mask = cv2.fillPoly(mask, [poly_corners], 1)
    return mask

def find_edges(mask) -> tuple:
    """Find the bounding box [x_min, x_max, y_min, y_max] of a binary mask."""
    
    # Find nonzero pixel locations
    rows = np.any(mask, axis=1)  # Check where rows have nonzero pixels
    cols = np.any(mask, axis=0)  # Check where columns have nonzero pixels

    if not np.any(rows) or not np.any(cols):
        return None  # No foreground pixels

    # Get min/max indices
    x_min, x_max = np.where(rows)[0][[0, -1]]
    y_min, y_max = np.where(cols)[0][[0, -1]]

    return np.array([x_min, x_max, y_min, y_max], dtype=np.int32)

import numpy as np

def find_intersect(edges, coord_0, coord_1):

    intersections = []
    limits = np.array(edges, dtype=np.float32)  # Ensure limits is a NumPy array to avoid mutability issues

    inside_0 = (edges[0] <= coord_0[0] <= edges[1]) and (edges[2] <= coord_0[1] <= edges[3])
    inside_1 = (edges[0] <= coord_1[0] <= edges[1]) and (edges[2] <= coord_1[1] <= edges[3])

    if inside_0:
        intersections.append(coord_0)
        if inside_1:
            return intersections  # No need to check intersections

    # Compute bounding box for the line segment
    limits[0] = np.min([coord_0[0], coord_1[0]]) - 1e-3 if edges[0] < np.min([coord_0[0], coord_1[0]]) < edges[1] else edges[0]
    limits[1] = np.max([coord_0[0], coord_1[0]]) + 1e-3 if edges[0] < np.max([coord_0[0], coord_1[0]]) < edges[1] else edges[1]
    limits[2] = np.min([coord_0[1], coord_1[1]]) - 1e-3 if edges[2] < np.min([coord_0[1], coord_1[1]]) < edges[3] else edges[2]
    limits[3] = np.max([coord_0[1], coord_1[1]]) + 1e-3 if edges[2] < np.max([coord_0[1], coord_1[1]]) < edges[3] else edges[3]

    dx = coord_1[0] - coord_0[0]
    dy = coord_1[1] - coord_0[1]

    vertical = np.abs(dx) < 1e-9
    horizontal = np.abs(dy) < 1e-9

    # Check vertical intersections
    if not vertical:
        intersect_0 = coord_0 + ((edges[0] - coord_0[0]) / dx) * np.array([dx, dy])
        intersect_1 = coord_0 + ((edges[1] - coord_0[0]) / dx) * np.array([dx, dy])

        if limits[0] <= intersect_0[0] <= limits[1] and limits[2] <= intersect_0[1] <= limits[3]:
            intersections.append(intersect_0)

        if limits[0] <= intersect_1[0] <= limits[1] and limits[2] <= intersect_1[1] <= limits[3]:
            intersections.append(intersect_1)

    # Check horizontal intersections
    if not horizontal:
        intersect_2 = coord_0 + ((edges[2] - coord_0[1]) / dy) * np.array([dx, dy])
        intersect_3 = coord_0 + ((edges[3] - coord_0[1]) / dy) * np.array([dx, dy])

        if limits[0] <= intersect_2[0] <= limits[1] and limits[2] <= intersect_2[1] <= limits[3]:
            intersections.append(intersect_2)

        if limits[0] <= intersect_3[0] <= limits[1] and limits[2] <= intersect_3[1] <= limits[3]:
            intersections.append(intersect_3)

    return np.array(intersections, dtype=np.int32)



def order_points_clockwise(points):
    """
    Orders points in clockwise order based on their centroid.
    :param points: (n,2) NumPy array of polygon points
    :return: Ordered (n,2) NumPy array
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Sort points based on angle from centroid
    def angle(p):
        return np.arctan2(p[1] - centroid[1], p[0] - centroid[0])

    sorted_points = sorted(points, key=angle)
    return np.array(sorted_points, dtype=np.int32)

        
        
def prepare_pixel_data(HSI_patch: np.ndarray, RGB_patch: np.ndarray, HSI_mask: np.ndarray, RGB_mask: np.ndarray)->tuple:
    """Used when patches partially overlap

    Args:
        HSI_patch (np.ndarray): Square HSI patch
        RGB_patch (np.ndarray): Square RGB patch
        HSI_mask (np.ndarray): HSI overlay mask
        RGB_mask (np.ndarray): RGB overlay mask

    Returns:
        tuple: HSI_data_flattened, RGB_data_flattened
    """

    skipped = 0
    HSI_output = np.zeros(shape=(np.sum(HSI_mask),HSI_patch.shape[2]))
    RGB_output = np.zeros(shape=(np.sum(RGB_mask),RGB_patch.shape[2]))
    for i in range(HSI_mask.shape[0]):
        for j in range(HSI_mask.shape[1]):
            if HSI_mask[i,j] == 0:
                skipped += 1
                break
            HSI_output[i+j*HSI_patch.shape[0]-skipped,:] = HSI_patch[i,j,:]
    skipped = 0
    for i in range(RGB_mask.shape[0]):
        for j in range(RGB_mask.shape[1]):
            if RGB_mask[i,j] == 0:
                skipped += 1
                break
            RGB_output[i+j*RGB_patch.shape[0]-skipped,:] = RGB_patch[i,j,:]
    
    return HSI_output, RGB_output

def rebuild_data(data, mask):
    patch = np.zeros(shape=(mask.shape[0], mask.shape[1], data.shape[1]))
    skipped = 0
    for i in range(data.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == 0:
                skipped += 1
                break

            patch[i,j] = data[i+j*mask.shape[0]-skipped,:]
    return patch
        
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
    