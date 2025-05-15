import cv2
import matplotlib.pyplot as plt
import numpy as np

class spatial_transform:
    def __init__(self, hsi_image, rgb_image):
        self.rgb_image = rgb_image #note that these images are grayscale representations
        self.hsi_image = hsi_image #note that these images are grayscale representations
        self.hr_transform, self.rh_transform = None, None
        self.hsi_limits = None
        self.write = True #output images
        self.rgb_mask, self.hsi_mask = None, None
        self.align_and_overlay()
        self.find_overlap()
        self.crop_hsi()
        self.align_and_overlay() #find transform after cropping
        self.find_overlap()
    
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
        self.hsi_limits = np.array([x_min, x_max, y_min, y_max]) 

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