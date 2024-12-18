"""
This script contains the code to detect ArUco 
markers and estimate their poses.
"""


from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2


class ArUcoDetector:
    """
    This class handles the detection of 
    ArUco markers and their pose estimation.
    """
    def __init__(self,
                 aruco_dict_type=cv2.aruco.DICT_6X6_250):
        """
        Initialize the detector with the specified ArUco 
        dictionary type.

        Input:
            - aruco_dict_type: Type of ArUco dictionary
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            aruco_dict_type
        )
        parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            aruco_dict, parameters
        )
        # Resized image
        self.resized_img = None

    def detect_markers(self):
        """
        This method detects ArUco markers in 
        the input image.

        Output:
            - Detected marker corners, IDs, and 
              rejected points
        """
        if self.resized_img is None:
            raise ValueError("Resized image is not set yet.")

        gray = cv2.cvtColor(self.resized_img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = \
            self.aruco_detector.detectMarkers(gray)
        return (
            corners,
            ids,
            rejected_img_points
        )

    def estimate_pose(self,
                      corner,
                      marker_points,
                      intrinsic_matrix,
                      dist_coeffs):
        """
        This method is a helper function to estimate pose for 
        a single marker.

        Input:
            - corner: Detected marker corner
            - marker_points: 3D coordinates of marker point
            - intrinsic_matrix: Camera intrinsic parameters
            - dist_coeffs: Camera distortion coefficients

        Output:
            - Rotation and translation vectors (rvec, tvec)
        """
        _, rvec, tvec = cv2.solvePnP(
            marker_points,
            corner,
            intrinsic_matrix,
            dist_coeffs
        )
        return rvec, tvec

    def estimate_pose_multiple(self,
                               corners,
                               marker_length,
                               intrinsic_matrix,
                               distortion_vector):
        """
        This method estimates the poses of multiple markers.

        Input:
            - corners: Detected marker corners
            - marker_length: Physical length of the marker 
                  in meters
            - intrinsic_matrix: Camera intrinsic parameters
            - distortion_vector: Camera distortion coefficients
        
        Output:
            - Rotation and translation vectors for each 
              marker
        """
        # Define the 3D coordinates of the marker points.
        marker_points = np.array([
            [[0, 0, 0]],  # Top-left corner as the origin
            [[0, marker_length, 0]],  # Bottom-left
            [[marker_length, marker_length, 0]],  # Bottom-right
            [[marker_length, 0, 0]]  # Top-right
        ], dtype=np.float32)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda corner: self.estimate_pose(
                    corner,
                    marker_points,
                    intrinsic_matrix,
                    distortion_vector
                ),
                corners
            ))
        rvecs, tvecs = zip(*results)

        return rvecs, tvecs

    def draw_axes(self,
                  rvec,
                  tvec,
                  intrinsic_matrix,
                  distortion_vector,
                  axis_length=0.05):
        """
        This method draws coordinate axes on the 
        image for the given pose.

        Input:
            - img: Input image
            - rvec: Rotation vector
            - tvec: Translation vector
            - camera: Camera object
            - axis_length: Length of the axes to be drawn
        """
        # Reshape rvec and tvec to ensure compatibility.
        rvec = np.array(rvec, dtype=np.float32).reshape(1, 3)
        tvec = np.array(tvec, dtype=np.float32).reshape(1, 3)

        # Draw axes.
        cv2.drawFrameAxes(
            self.resized_img,
            intrinsic_matrix,
            distortion_vector,
            rvec,
            tvec,
            axis_length
        )

        return

    def draw_detected_markers(self,
                              corners):
        """
        This method draws detected markers 
        on the image.

        Input:
            - corners: Detected marker corners
        """
        # Draw detected markers on the resized image.
        cv2.aruco.drawDetectedMarkers(
            self.resized_img,
            corners
        )
        return
    