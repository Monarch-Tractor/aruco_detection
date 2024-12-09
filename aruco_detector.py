"""
This script runs ArUco marker detection and pose 
estimation on an input image.
"""


import cv2
import numpy as np

from camera import Camera


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

    def detect_markers(self,
                       img):
        """
        This method detects ArUco markers in 
        the input image.
        Input:
            - img: Input image

        Output:
            - Detected marker corners, IDs, and 
              rejected points
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = \
            self.aruco_detector.detectMarkers(gray)
        return (
            corners,
            ids,
            rejected_img_points
        )

    def estimate_pose(self,
                      corners,
                      marker_length,
                      camera):
        """
        This method estimates the pose of detected markers.

        Input:
            - corners: Detected marker corners
            - marker_length: Physical length of the marker 
                  in meters
            - camera: Camera object with intrinsic and 
                  distortion parameters
        
        Output:
            - Rotation and translation vectors for each 
                  marker
        """
        rvecs = []
        tvecs = []
        marker_points = np.array([
            [[0, 0, 0]],  # Top-left corner as the origin
            [[0, marker_length, 0]],  # Bottom-left
            [[marker_length, marker_length, 0]],  # Bottom-right
            [[marker_length, 0, 0]]  # Top-right
        ], dtype=np.float32)
        for corner in corners:
            # Estimate the pose using solvePnP.
            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                corner,
                camera.intrinsic_matrix,
                camera.dist_coeffs

            )
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs

    def draw_axes(self,
                  img,
                  rvec,
                  tvec,
                  camera,
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
        rvec = np.array(
            rvec,
            dtype=np.float32
        ).reshape(1, 3)
        tvec = np.array(
            tvec,
            dtype=np.float32
        ).reshape(1, 3)

        # Draw axes using cv2.drawFrameAxes.
        cv2.drawFrameAxes(
            img,
            camera.intrinsic_matrix,
            camera.dist_coeffs,
            rvec,
            tvec,
            axis_length
        )

    def draw_detected_markers(self,
                              img,
                              corners):
        """
        This method draws detected markers 
        on the image.

        Input:
            - img: Input image
            - corners: Detected marker corners
        """
        # Draw detected markers on the resized image.
        cv2.aruco.drawDetectedMarkers(
            img,
            corners
        )
        return

class ArUcoProcessor:
    """
    This method orchestrates the detection and 
    pose estimation of ArUco markers.
    """
    def __init__(self,
                 camera,
                 marker_length=0.05):
        """
        This method initializes the processor with 
        camera and marker length.

        Input:
            - camera: Camera object
            - marker_length: Physical length of the marker 
                  in meters
        """
        self.camera = camera
        self.marker_length = marker_length
        self.detector = ArUcoDetector()

    def process_image(self,
                      img):
        """
        This metod detects markers, estimate poses, 
        and overlay results on the image.

        Input:
            - img: Input image

        Output:
            - Processed image with detected markers and 
              axes overlay
        """
        # Resize the image to 1280 x 720.
        resized_img = cv2.resize(img, (1280, 720))

        # Detect markers in the resized image.
        corners, ids, _ = \
            self.detector.detect_markers(resized_img)

        # Print the detected markers
        print("Detected markers:", ids)

        if ids is not None:
            # Draw detected markers on the resized image.
            self.detector.draw_detected_markers(
                img=resized_img,
                corners=corners
            )

            # Estimate poses of detected markers.
            rvecs, tvecs = self.detector.estimate_pose(
                corners=corners,
                marker_length=self.marker_length,
                camera=self.camera
            )

            # Draw axes for each detected marker.
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                self.detector.draw_axes(
                    img=resized_img,
                    rvec=rvec,
                    tvec=tvec,
                    camera=self.camera
                )

                # Draw the marker ID below the detected
                # marker.
                # Get the first corner for positioning
                # the text.
                corner = corners[i][0]
                # Taking the top-left corner for text
                # placement.
                x, y = int(corner[0][0]), int(corner[0][1])
                cv2.putText(
                    resized_img,
                    str(ids[i][0]),
                    (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )


        # Return the resized and processed image.
        return resized_img


# Main script
if __name__ == "__main__":
    # Camera name
    CAMERA_NAME = "zed_front"

    # Camera information ROS topic
    CAMERA_INFO_TOPIC = f"/{CAMERA_NAME}/camera_info"

    # Initialize camera and processor.
    camera = Camera(
        camera_name=CAMERA_NAME,
        camera_info_topic=CAMERA_INFO_TOPIC,
        use_default_intrinsics=True
    )
    processor = ArUcoProcessor(
        camera=camera
    )

    # Load the input image
    IN_IMG_PATH = "aruco_grid-in.png"
    OUT_IMG_PATH = "aruco_grid-out.png"
    in_img = cv2.imread(IN_IMG_PATH)

    # Process the image.
    processed_img = processor.process_image(in_img)

    # Display and save the results.
    cv2.imshow(
        "ArUco Marker Detection and Pose Estimation",
        processed_img
    )
    cv2.imwrite(OUT_IMG_PATH, processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
