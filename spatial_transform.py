import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import utilities as util

class spatial_transform:
    def __init__(self, hsi_image, rgb_image, write=True, debug = False):
        self.rgb_image = rgb_image #note that these images are grayscale representations
        self.hsi_image = hsi_image #note that these images are grayscale representations
        self.hr_transform, self.rh_transform = None, None
        self.hsi_mask, self.rgb_mask = None, None
        self.hsi_limits = None
        self.write = write
        self.debug = debug
        self.gaussian_table = create_gaussian_array()
        self.align_and_overlay()
        self.find_overlap()
        if self.debug:
            print(f"hsi mask shape: {self.hsi_mask.shape}\n rgb mask shape: {self.rgb_mask.shape}")
            # Create a side-by-side visualization
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(self.hsi_mask, cmap='gray')
            axes[0].set_title('h_mask (Pixel to RGB)')

            axes[1].imshow(self.rgb_mask, cmap='gray')
            axes[1].set_title('r_mask (RGB to Pixel)')

            # Remove axis labels for better clarity
            for ax in axes:
                ax.axis('off')

            plt.show()
        assert np.any(self.hsi_mask) and np.any(self.rgb_mask), "Mask generation failed"
        self.crop_hsi()
        self.align_and_overlay() #find transform after cropping
        self.find_overlap()
        if self.debug:
            print(f"hsi mask shape: {self.hsi_mask.shape}\n rgb mask shape: {self.rgb_mask.shape}")
            # Create a side-by-side visualization
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(self.hsi_mask, cmap='gray')
            axes[0].set_title('h_mask (Pixel to RGB)')

            axes[1].imshow(self.rgb_mask, cmap='gray')
            axes[1].set_title('r_mask (RGB to Pixel)')

            # Remove axis labels for better clarity
            for ax in axes:
                ax.axis('off')

            plt.show()
    
    def align_and_overlay(self):
        rgb_image = cv2.GaussianBlur(self.rgb_image, (15, 15), 0)
        hsi_image = self.hsi_image

        if self.write:
            cv2.imwrite("output/rgb.png", rgb_image)
            cv2.imwrite("output/hsi.png", self.hsi_image)
        
        # Initialize SIFT detector - parameters are almost definitely sub-optimal
        sift = cv2.SIFT_create(nfeatures=50000, contrastThreshold=0.015, edgeThreshold=40, sigma=2)

        # Detect keypoints and descriptors
        kp_hsi, des_hsi = sift.detectAndCompute(hsi_image, None)
        kp_rgb, des_rgb = sift.detectAndCompute(rgb_image, None)

        # Use BFMatcher to find the best matches
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des_hsi, des_rgb, k=2)

        # Apply Lowe's ratio test to filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        img_matches = cv2.drawMatches(hsi_image, kp_hsi, rgb_image, kp_rgb, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if self.write:
            plt.figure(figsize=(20, 10))
            plt.imshow(img_matches)
            
            plt.savefig(f"output/matches.png")
            plt.close()

        # Extract the matching keypoints
        src_pts = np.float32([kp_hsi[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5,confidence=0.995, maxIters=5000)
        N, mask2 = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransacReprojThreshold=5,confidence=0.995, maxIters=5000)

        if self.write:
            # Transform
            aligned_hsi = cv2.warpPerspective(hsi_image, M, (rgb_image.shape[1], rgb_image.shape[0]))
            aligned_rgb = cv2.warpPerspective(rgb_image, N, (hsi_image.shape[1], hsi_image.shape[0]))
            # Save the aligned image
            cv2.imwrite(f"output/aligned_hsi.png", aligned_hsi)
            cv2.imwrite(f"output/aligned_rgb.png", aligned_rgb)

            # blend images
            overlay_img = cv2.addWeighted(hsi_image, 0.5, aligned_rgb, 0.5, 0)
            
            cv2.imwrite(f"output/overlay.png", overlay_img)
        self.hr_transform = M
        self.rh_transform = N
        
    def find_overlap(self):
        """Generates h_mask and r_mask

        Args:
            direction (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        A_points = [0, self.hsi_image.shape[1]-1, 0, self.hsi_image.shape[0]-1]
        B_points = [0, self.rgb_image.shape[1]-1, 0, self.rgb_image.shape[0]-1]
        
        A_corners = np.array([
            [A_points[0], A_points[2], 1.0], #x0, y0
            [A_points[1], A_points[2], 1.0], #x1, y0
            [A_points[1], A_points[3], 1.0], #x1, y1
            [A_points[0], A_points[3], 1.0]  #x0, y1
        ])
        B_corners = np.array([
            [B_points[0], B_points[2], 1.0], #x0, y0
            [B_points[1], B_points[2], 1.0], #x1, y0
            [B_points[1], B_points[3], 1.0], #x1, y1
            [B_points[0], B_points[3], 1.0]  #x0, y1
        ])

        B_corners_warped_homogeneous = (self.rh_transform @ B_corners.T).T
        B_corners_warped = (B_corners_warped_homogeneous[:, :2] / B_corners_warped_homogeneous[:, 2:3]).astype(np.int32)
        
        B_corners_warped = order_points_clockwise(B_corners_warped)
        
        A_corners_warped_homogeneous = (self.hr_transform @ A_corners.T).T
        A_corners_warped = (A_corners_warped_homogeneous[:, :2] / A_corners_warped_homogeneous[:, 2:3]).astype(np.int32)
        
        A_corners_warped = order_points_clockwise(A_corners_warped)
        
        stacked_points_hsi = np.vstack([B_corners_warped, A_corners[:,:2]])
        stacked_points_rgb = np.vstack([A_corners_warped, B_corners[:,:2]])
        
        adjust_x_hsi = -np.min(stacked_points_hsi[:,0]).astype(np.int32) #find negative extent of corners
        adjust_y_hsi = -np.min(stacked_points_hsi[:,1]).astype(np.int32)
        
        width_hsi = (np.max(stacked_points_hsi[:,0]) - np.min(stacked_points_hsi[:,0])).astype(np.int32) #Mask dimensions to fit entire projection
        height_hsi = (np.max(stacked_points_hsi[:,1]) - np.min(stacked_points_hsi[:,1])).astype(np.int32)
        assert width_hsi < 20000 and height_hsi < 20000, f"Projection error"
        B_corners_warped[:] += [int(adjust_x_hsi), int(adjust_y_hsi)]
        
        mask_ext_hsi = np.zeros(shape=(height_hsi, width_hsi), dtype=np.uint8)
        mask_ext_hsi = cv2.fillPoly(mask_ext_hsi, [B_corners_warped], 1)
        self.hsi_mask = mask_ext_hsi[adjust_y_hsi:adjust_y_hsi+A_points[3], adjust_x_hsi:adjust_x_hsi+A_points[1]].copy()
        
        adjust_x_rgb = -np.min(stacked_points_rgb[:,0]).astype(np.int32) #find negative extent of corners
        adjust_y_rgb = -np.min(stacked_points_rgb[:,1]).astype(np.int32)
        
        width_rgb = (np.max(stacked_points_rgb[:,0]) - np.min(stacked_points_rgb[:,0])).astype(np.int32) #Mask dimensions to fit entire projection
        height_rgb = (np.max(stacked_points_rgb[:,1]) - np.min(stacked_points_rgb[:,1])).astype(np.int32)
        
        assert 0 < width_rgb < 20000 and 0 < height_rgb < 20000, f"Projection error rgb width,height: {width_rgb,height_rgb}"
        
        A_corners_warped[:] += [adjust_x_rgb, adjust_y_rgb]
        
        mask_ext_rgb = np.zeros(shape=(height_rgb, width_rgb), dtype=np.uint8)
        mask_ext_rgb = cv2.fillPoly(mask_ext_rgb, [A_corners_warped], 1)
        self.rgb_mask = mask_ext_rgb[adjust_y_rgb:adjust_y_rgb+B_points[3], adjust_x_rgb:adjust_x_rgb+B_points[1]].copy()
        
    def crop_hsi(self):
        """Find the bounding box [x_min, x_max, y_min, y_max] of a binary mask."""
    
        # Find nonzero pixel locations
        rows = np.any(self.hsi_mask, axis=1)  # Check where rows have nonzero pixels
        cols = np.any(self.hsi_mask, axis=0)  # Check where columns have nonzero pixels

        # Get min/max indices
        x_min, x_max = np.where(rows)[0][[0, -1]]
        y_min, y_max = np.where(cols)[0][[0, -1]]

        self.hsi_image = self.hsi_image[x_min:x_max,y_min:y_max].copy()
        self.hsi_limits = np.array([x_min, x_max, y_min, y_max]) #needed by fusion module
        
    def get_mask_subset(self, rgb_limits) -> tuple:
        """_summary_

        Args:
            rgb_limits (np.ndarray): [x_min, x_max, y_min, x_max] defining area of interest for rgb image

        Returns:
            tuple: rgb_mask, hsi_mask, hsi_limits for area of interest
        """
        rgb_mask_subset = self.rgb_mask[rgb_limits[0]:rgb_limits[1],rgb_limits[2]:rgb_limits[3]]
        rgb_points = np.array([
            [rgb_limits[0], rgb_limits[2], 1.0], #x0, y0
            [rgb_limits[1], rgb_limits[2], 1.0], #x1, y0
            [rgb_limits[1], rgb_limits[3], 1.0], #x1, y1
            [rgb_limits[0], rgb_limits[3], 1.0]  #x0, y1        
            ])
        hsi_points_homogenous = (self.rh_transform@rgb_points.T.astype(np.float32)).T
        hsi_points = (hsi_points_homogenous[:, :2] / hsi_points_homogenous[:, 2:3])
        
        hsi_limits = np.array([np.clip(np.min(hsi_points[:,0]),0,self.hsi_image.shape[1]), 
                               np.clip(np.max(hsi_points[:,0]+1),0,self.hsi_image.shape[1]),
                               np.clip(np.min(hsi_points[:,1]),0,self.hsi_image.shape[0]), 
                               np.clip(np.max(hsi_points[:,1]+1),0,self.hsi_image.shape[0]),
                                ]).astype(np.int32)
        
        hsi_mask_subset = self.hsi_mask[hsi_limits[0]:hsi_limits[1],hsi_limits[2]:hsi_limits[3]]
        print(f"hsi_mask_sum: {np.sum(hsi_mask_subset)}\nrgb_mask_sum: {np.sum(rgb_mask_subset)}")
        
        print(f"hsi mask shape: {hsi_mask_subset.shape}\n rgb mask shape: {rgb_mask_subset.shape}")
        # Create a side-by-side visualization
        """fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(hsi_mask_subset, cmap='gray')
        axes[0].set_title('h_mask (Pixel to RGB)')

        axes[1].imshow(rgb_mask_subset, cmap='gray')
        axes[1].set_title('r_mask (RGB to Pixel)')

        # Remove axis labels for better clarity
        for ax in axes:
            ax.axis('off')

        plt.show()"""
        return rgb_mask_subset, hsi_mask_subset, hsi_limits
    
    def get_pixels2(self, rgb_limits):
        debug = True
        rgb_x, rgb_y = rgb_limits[1]-rgb_limits[0], rgb_limits[3]-rgb_limits[2]
        rgb_pixels = rgb_x*rgb_y
        
        x_min, x_max, y_min, y_max = rgb_limits

        # Generate pixel centers for x and y
        x_coords = np.arange(x_min, x_max) + 0.5
        y_coords = np.arange(y_min, y_max) + 0.5

        # Create meshgrid and flatten in row-major order
        x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')

        # Stack into (N, 3) where each row is [x, y, 1]
        full_coords = np.column_stack([x_grid.ravel(), y_grid.ravel(), np.ones(x_grid.size)])
        hsi_centers = (self.rh_transform@full_coords.T).T
        hsi_centers = hsi_centers[:,:2] / hsi_centers[:,2:3]
        
        hsi_limits = np.array([np.floor(np.min(hsi_centers[:,0])-1),
                               np.ceil(np.max(hsi_centers[:,0])+1),
                               np.floor(np.min(hsi_centers[:,1])-1),
                               np.ceil(np.max(hsi_centers[:,1])+1),
                               ]).astype(np.int32)

        if (hsi_limits[0] < 0 or hsi_limits[2] < 0 or
            hsi_limits[1] > self.hsi_mask.shape[0] or
            hsi_limits[3] > self.hsi_mask.shape[1]):
            return None, None
        
        hsi_x, hsi_y = hsi_limits[1]-hsi_limits[0], hsi_limits[3]-hsi_limits[2]
        hsi_indices = hsi_centers.astype(np.int32)-np.array([hsi_limits[0], hsi_limits[2]],dtype=np.int32)
        
        hsi_pixels = hsi_x*hsi_y
        spatial_transform = np.zeros(shape=(hsi_pixels, rgb_pixels))
        
        for i in range(hsi_indices.shape[0]):
            rgb_coords = full_coords[i,:2].astype(np.uint32)
            hsi_distribution = self.get_distribution(hsi_centers[i])
            for k in range(hsi_distribution.shape[0]):
                for l in range(hsi_distribution.shape[1]):
                    hsi_coords = hsi_indices[i] + np.array([k-1, l-1])
                    assert 0 <= hsi_coords[0] <= hsi_x and 0 <= hsi_coords[1] <= hsi_y, f"attempting to write to unmapped HSI pixel\nhsi_coords: {hsi_coords}\nhsi_limits: {hsi_limits}"
                    rgb_index = (rgb_coords[0]-rgb_limits[0]) + (rgb_coords[1]-rgb_limits[2])*rgb_x
                    hsi_index = hsi_coords[0] + hsi_coords[1]*hsi_x
                    spatial_transform[hsi_index, rgb_index] = hsi_distribution[k,l]
        return spatial_transform, hsi_limits
        
        
    """def get_pixels(self, rgb_mask, hsi_mask, rgb_origin, hsi_origin): #THIS IS TERRIBLE CODE
        
        rgb_pixels = np.sum(rgb_mask)
        hsi_pixels = np.sum(hsi_mask)
        
        spatial_transform = np.zeros(shape=(hsi_pixels, rgb_pixels))
        hsi_skipped = np.zeros(shape=hsi_mask.shape[0]*hsi_mask.shape[1])
        skipped = 0
        for i in range(hsi_mask.shape[0]):
            for j in range(hsi_mask.shape[1]):
                if hsi_mask[i,j] == 0:
                    skipped += 1
                    hsi_skipped[i+j*hsi_mask.shape[0]] = skipped
        skipped = 0
        
        for i in range(rgb_mask.shape[0]):
            for j in range(rgb_mask.shape[1]):
                if rgb_mask[i,j] == 1:
                    rgb_coords = np.array([i+rgb_origin[0]+0.5, j+rgb_origin[1]+0.5, 1.0])
                    hsi_coords_homogenous = (self.rh_transform@rgb_coords.T).T
                    hsi_distribution = self.get_distribution(hsi_coords_homogenous[:2])
                    hsi_coords = np.round((hsi_coords_homogenous[:2] / hsi_coords_homogenous[2:3]) - hsi_origin, decimals=0).astype(np.int32)
                    rgb_index = i+j*rgb_mask.shape[0]-skipped
                    for k in range(hsi_distribution.shape[0]):
                        for l in range(hsi_distribution.shape[1]):
                            hsi_x = hsi_coords[0]+k-2
                            hsi_y = hsi_coords[1]+l-2
                            hsi_index = hsi_x+hsi_y*hsi_mask.shape[0]
                            if hsi_mask[hsi_x,hsi_y] == 1:
                                hsi_index -= hsi_skipped[hsi_index]
                                assert hsi_index < spatial_transform.shape[0], f"===SPATIAL TRANSFORM===\nhsi coords: {hsi_x, hsi_y}\n hsi index: {hsi_index}\nSpatial transform: {spatial_transform.shape}"
                                spatial_transform[hsi_index, rgb_index] = hsi_distribution[k,l]
                    
                else:
                    skipped += 1
            
            
            
        #assert np.all(np.any(spatial_transform, axis=0)), "Unused HSI pixel in transform"
        normalization_weights = np.sum(spatial_transform, axis=0)
        spatial_transform = spatial_transform/normalization_weights
        
        return spatial_transform"""

    def get_distribution(self, coords):
        hsi_rem = np.remainder(coords, 1.0)
        assert np.all((0.0 <= hsi_rem) & (hsi_rem < 1.0)), f"Error getting pixel distribution, hsi_rem: {hsi_rem}"
        dx_0, dy_0 = np.clip(0.5-hsi_rem[0], a_min=0, a_max=0.5), np.clip(0.5-hsi_rem[1], a_min=0, a_max=0.5) #there is definitely a better way to do this
        dx_1, dy_1 = 1-hsi_rem[0]//0.5, 1-hsi_rem[1]//0.5
        dx_0_indices, dy_0_indices = np.array([0, np.round(dx_0, decimals=2)*100], dtype=np.int32), np.array([0, np.round(dy_0, decimals=2)*100], dtype=np.int32)
        dx_1_indices, dy_1_indices = np.array([np.round(dx_0, decimals=2)*100, np.round(dx_1, decimals=2)*100], dtype=np.int32), np.array([np.round(dy_0, decimals=2)*100, np.round(dy_1, decimals=2)*100], dtype=np.int32)
        dx_2_indices, dy_2_indices = np.array([np.round(dx_1, decimals=2)*100, 100], dtype=np.int32), np.array([np.round(dy_1, decimals=2)*100, 100], dtype=np.int32)
        distribution = np.zeros(shape=(3,3))
        distribution[0,0] = np.sum(self.gaussian_table[dx_0_indices[0]:dx_0_indices[1],dy_0_indices[0]:dy_0_indices[1]])
        distribution[1,0] = np.sum(self.gaussian_table[dx_1_indices[0]:dx_1_indices[1],dy_0_indices[0]:dy_0_indices[1]])
        distribution[2,0] = np.sum(self.gaussian_table[dx_2_indices[0]:dx_2_indices[1],dy_0_indices[0]:dy_0_indices[1]])
        distribution[0,1] = np.sum(self.gaussian_table[dx_0_indices[0]:dx_0_indices[1],dy_1_indices[0]:dy_1_indices[1]])
        distribution[1,1] = np.sum(self.gaussian_table[dx_1_indices[0]:dx_1_indices[1],dy_1_indices[0]:dy_1_indices[1]])
        distribution[2,1] = np.sum(self.gaussian_table[dx_2_indices[0]:dx_2_indices[1],dy_1_indices[0]:dy_1_indices[1]])
        distribution[0,2] = np.sum(self.gaussian_table[dx_0_indices[0]:dx_0_indices[1],dy_2_indices[0]:dy_2_indices[1]])
        distribution[1,2] = np.sum(self.gaussian_table[dx_1_indices[0]:dx_1_indices[1],dy_2_indices[0]:dy_2_indices[1]])
        distribution[2,2] = np.sum(self.gaussian_table[dx_2_indices[0]:dx_2_indices[1],dy_2_indices[0]:dy_2_indices[1]])
        
        return distribution
    
    def get_extent(self, limits):
        RGB_points = np.array([[limits[0], limits[2], 1.0],
                           [limits[1], limits[2], 1.0],
                           [limits[1], limits[3], 1.0],
                           [limits[0], limits[3], 1.0],
                           ])
        #print(f"rgb points: {RGB_points}")
        #print(f"transform: {self.rh_transform}")
        HSI_points = (self.rh_transform@RGB_points.T).T
        #print(f"HSI points homogeneous: {HSI_points}")
        HSI_points = (HSI_points[:,:2] / HSI_points[:,2:3]).astype(np.int32)
        #print(f"HSI points: {HSI_points}")
        hsi_limits = np.clip(np.array([ #format [x_min, x_max, y_min, y_max]
                                    np.floor(np.min(HSI_points[:,0])).astype(np.int32) - 1,
                                    np.ceil(np.max(HSI_points[:,0])).astype(np.int32) + 1,
                                    np.floor(np.min(HSI_points[:,1])).astype(np.int32) - 1,
                                    np.ceil(np.max(HSI_points[:,1])).astype(np.int32) + 1,
                                    ]),
                                a_min=np.array([0, 0, 0, 0]),
                                a_max=np.array([self.hsi_mask.shape[0], self.hsi_mask.shape[0],
                                                self.hsi_mask.shape[1], self.hsi_mask.shape[1]])
                            )
        #print(f"HSI limits: {hsi_limits}")
        return hsi_limits
        
    
def remove_unused(transform, array):
    mask = np.any(transform != 0, axis=1)  # Boolean mask for non-zero rows
    #print(f"mask shape: {mask.shape}\narray shape: {array.shape}\ntransform shape: {transform.shape}")
    new_transform = transform[mask]  # Efficient row filtering
    new_array = array.T[mask].T
    
    assert not np.any(np.sum(new_transform, axis=1) == 0), "unmapped hsi pixels"
    
    return new_transform, new_array

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
    print(f"hsi mask shape: {HSI_mask.shape}\n rgb mask shape: {RGB_mask.shape}")
    # Create a side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(HSI_mask, cmap='gray')
    axes[0].set_title('h_mask (Pixel to RGB)')

    axes[1].imshow(RGB_mask, cmap='gray')
    axes[1].set_title('r_mask (RGB to Pixel)')

    # Remove axis labels for better clarity
    for ax in axes:
        ax.axis('off')

    plt.show()

    skipped = 0
    HSI_output = np.zeros(shape=(np.sum(HSI_mask),HSI_patch.shape[2]))
    RGB_output = np.zeros(shape=(np.sum(RGB_mask),RGB_patch.shape[2]))
    print(HSI_output.shape)
    print(RGB_output.shape)
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
    