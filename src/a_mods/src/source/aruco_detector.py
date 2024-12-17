"""
This script defines the ArUco processor which runs
the marker detection and pose estimation algorithm 
on any input image.
"""


from concurrent.futures import (
    ThreadPoolExecutor,
    wait
)

import numpy as np
import cv2

import tf2_ros
import tf2_geometry_msgs

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose

from .camera import Camera


def rvec_tvec_to_pose(rvec, tvec):
    """
    This method converts the rotation and translation 
    vectors to a pose message.

    Input:
        - rvec: Rotation vector
        - tvec: Translation vector

    Output:
        - Pose message
    """
    # Convert rvec to quaternion
    rotation = R.from_rotvec(rvec.flatten())
    quaternion = rotation.as_quat()

    # Create the pose message
    pose = Pose()
    pose.position.x = tvec[0]
    pose.position.y = tvec[1]
    pose.position.z = tvec[2]
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose


def transform_object_to_global(rvec_camera,
                               tvec_camera,
                               rvec_object_camera,
                               tvec_object_camera):
    """
    This method transforms the object pose from the 
    camera frame to the global frame.
    """
    # Convert rvecs to rotation matrices
    R_camera, _ = cv2.Rodrigues(rvec_camera)
    R_object_camera, _ = cv2.Rodrigues(rvec_object_camera)

    # Compute global rotation
    R_object_global = R_camera @ R_object_camera

    # Compute global translation
    t_object_global = R_camera @ tvec_object_camera + tvec_camera

    # Convert rotation matrix back to rvec
    rvec_object_global, _ = cv2.Rodrigues(R_object_global)

    return (
        rvec_object_global,
        t_object_global
    )


def transform_camera_to_global(rvec_object_camera,
                               tvec_object_camera,
                               rvec_object_global,
                               tvec_object_global):
    """
    This method transforms the camera pose from the
    object frame to the global frame.
    """
    # Convert rvecs to rotation matrices
    R_object_camera, _ = cv2.Rodrigues(rvec_object_camera)
    R_object_global, _ = cv2.Rodrigues(rvec_object_global)

    # Compute camera rotation
    R_camera = R_object_global.T @ R_object_camera

    # Compute camera translation
    t_camera = R_object_global.T @ (
        tvec_object_camera - tvec_object_global
    )

    # Convert rotation matrix back to rvec
    rvec_camera, _ = cv2.Rodrigues(R_camera)

    return rvec_camera, t_camera


def convert_rvec_tvec_to_pose(rvec, tvec):
    """
    This method converts the rotation and translation 
    vectors to a pose message.

    Input:
        - rvec: Rotation vector
        - tvec: Translation vector

    Output:
        - Pose message
    """
    # Convert rvec to quaternion
    rotation = R.from_rotvec(rvec.flatten())
    quaternion = rotation.as_quat()

    # Create the pose message
    pose = Pose()
    pose.position.x = tvec[0]
    pose.position.y = tvec[1]
    pose.position.z = tvec[2]
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]

    return pose


class ArUcoProcessor:
    """
    This class handles the detection of ArUco markers, 
    pose estimation, and camera pose estimation based 
    on the detected markers.
    """
    def __init__(self,
                 camera,
                 marker_length=0.05,
                 aruco_dict_type=cv2.aruco.DICT_6X6_250):
        """
        Initialize the processor with camera, marker length, 
        and ArUco dictionary type.

        Input:
            - camera: Camera object
            - marker_length: Physical length of the marker 
                  in meters
            - aruco_dict_type: Type of ArUco dictionary
        """
        self.camera = camera
        self.marker_length = marker_length

        # Initialize ArUco detector
        aruco_dict = \
            cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = \
            cv2.aruco.DetectorParameters()
        self.aruco_detector = \
            cv2.aruco.ArucoDetector(aruco_dict, parameters)

        # Resized image
        self.resized_img = None

        # Camera pose initialization
        self.camera_pose = Pose()
        self.camera_pose.position.x = 0
        self.camera_pose.position.y = 0
        self.camera_pose.position.z = 0
        self.camera_pose.orientation.x = 0
        self.camera_pose.orientation.y = 0
        self.camera_pose.orientation.z = 0
        self.camera_pose.orientation.w = 1

        # Relative and global poses
        self.relative_poses = {}
        self.global_poses = {}

    def detect_markers(self):
        """
        Detect ArUco markers in the input image.

        Output:
            - Detected marker corners, IDs, and rejected points
        """
        if self.resized_img is None:
            raise ValueError("Resized image is not set yet.")

        gray = cv2.cvtColor(self.resized_img, cv2.COLOR_BGR2GRAY)
        corners, ids, \
        rejected_img_points = self.aruco_detector.detectMarkers(gray)

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
        This method estimates the pose of a single marker.

        Input:
            - corner: Detected marker corner
            - marker_points: 3D coordinates of marker points
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

        return (rvec, tvec)

    def estimate_pose_multiple(self,
                               corners,
                               marker_length,
                               intrinsic_matrix,
                               distortion_vector):
        """
        This method estimates poses of multiple markers.

        Input:
            - corners: Detected marker corners
            - marker_length: Physical length of the marker in meters
            - intrinsic_matrix: Camera intrinsic parameters
            - distortion_vector: Camera distortion coefficients

        Output:
            - Rotation and translation vectors for each marker
        """
        marker_points = np.array([
            [[0, 0, 0]],  # Top-left corner
            [[0, marker_length, 0]],  # Bottom-left
            [[marker_length, marker_length, 0]],  # Bottom-right
            [[marker_length, 0, 0]]  # Top-right
        ], dtype=np.float32)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda corner: self.estimate_pose(
                    corner=corner,
                    marker_points=marker_points,
                    intrinsic_matrix=intrinsic_matrix,
                    dist_coeffs=distortion_vector
                ),
                corners
            ))

        rvecs, tvecs = zip(*results)
        return (rvecs, tvecs)

    def draw_axes(self,
                  rvec,
                  tvec,
                  intrinsic_matrix,
                  distortion_vector,
                  axis_length=0.05):
        """
        This method draws coordinate axes on the image 
        for the given pose.

        Input:
            - rvec: Rotation vector
            - tvec: Translation vector
            - intrinsic_matrix: Camera intrinsic parameters
            - distortion_vector: Camera distortion coefficients
            - axis_length: Length of the axes to be drawn
        """
        rvec = np.array(rvec, dtype=np.float32).reshape(1, 3)
        tvec = np.array(tvec, dtype=np.float32).reshape(1, 3)
        cv2.drawFrameAxes(
            self.resized_img,
            intrinsic_matrix,
            distortion_vector,
            rvec,
            tvec,
            axis_length
        )
        return

    def draw_detected_markers(self, corners):
        """
        Draw detected markers on the image.

        Input:
            - corners: Detected marker corners
        """
        cv2.aruco.drawDetectedMarkers(
            self.resized_img,
            corners
        )
        return

    def postprocess_marker_image(self,
                                 rvec,
                                 tvec,
                                 corner,
                                 marker_id):
        """
        This method post-processes a single marker by drawing 
        its axes and overlaying its ID.

        Input:
            - rvec: Rotation vector
            - tvec: Translation vector
            - corner: Marker corner (used for text placement)
            - marker_id: ID of the marker
        """
        self.draw_axes(
            rvec=rvec,
            tvec=tvec,
            intrinsic_matrix=self.camera.intrinsic_matrix,
            distortion_vector=self.camera.distortion_vector
        )
        x, y = int(corner[0][0]), int(corner[0][1])  # Top-left corner
        cv2.putText(
            self.resized_img,
            str(marker_id),
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (124, 214, 166),
            2,
            cv2.LINE_AA
        )
        return

    def process_image(self, img):
        """
        This method detect markers, estimates poses, and overlays 
        results on the image.

        Input:
            - img: Input image

        Output:
            - Processed image with detected markers and axes 
                  overlay
        """
        self.resized_img = cv2.resize(img, (1280, 720))
        corners, ids, _ = self.detect_markers()
        print("Detected markers:", ids)

        if ids is not None:
            self.draw_detected_markers(corners)
            rvecs, tvecs = self.estimate_pose_multiple(
                corners=corners,
                marker_length=self.marker_length,
                intrinsic_matrix=self.camera.intrinsic_matrix,
                distortion_vector=self.camera.distortion_vector
            )

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.postprocess_marker_image,
                        rvec,
                        tvec,
                        corners[i][0],
                        ids[i][0]
                    )
                    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs))
                ]
                wait(futures)

        self.relative_poses = {}
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            self.relative_poses[ids[i][0]] = (rvec, tvec)

        self.camera_pose = self.set_camera_pose_estimate()
        return self.resized_img

    def get_relative_poses(self):
        """
        This method returns the relative poses of the 
        detected markers.
        """
        relative_pose = None
        for (_, (rvec, tvec)) in self.relative_poses.items():
            relative_pose = rvec_tvec_to_pose(
                rvec=rvec,
                tvec=tvec
            )
            break
        return relative_pose

    def initialize_aruco_poses(self, c_rvec, c_tvec):
        """
        This method initializes the ArUco poses with 
        the given camera rotation and translation vectors.
        """
        for (id, (rvec, tvec)) in self.relative_poses.items():
            rvec_global, tvec_global = transform_object_to_global(
                rvec_camera=c_rvec,
                tvec_camera=c_tvec,
                rvec_object_camera=rvec,
                tvec_object_camera=tvec
            )
            self.global_poses[id] = (rvec_global, tvec_global)

    def get_global_poses(self):
        """
        This method returns the global poses of the 
        detected markers.
        """
        return self.global_poses

    def set_camera_pose_estimate(self):
        """
        This method returns the camera pose estimate based 
        on the detected ArUco markers.
        """
        ids = list(self.global_poses.keys())
        for id in ids:
            if id in self.relative_poses:
                rvec, tvec = self.relative_poses[id]
                c_rvec, c_tvec = transform_camera_to_global(
                    rvec_object_camera=rvec,
                    tvec_object_camera=tvec,
                    rvec_object_global=self.global_poses[id][0],
                    tvec_object_global=self.global_poses[id][1]
                )

        if len(ids) != 0:
            self.camera_pose = rvec_tvec_to_pose(
                rvec=c_rvec,
                tvec=c_tvec
            )

        return self.camera_pose

    def get_camera_pose_estimate(self):
        """
        This method returns the camera pose estimate.
        """
        return self.camera_pose


# Main driver code
if __name__ == "__main__":
    CAMERA_NAME = "zed_front"
    CAMERA_INFO_TOPIC = f"/{CAMERA_NAME}/camera_info"

    # Initialize camera and processor
    camera = Camera(
        camera_name=CAMERA_NAME,
        camera_info_topic=CAMERA_INFO_TOPIC,
        use_default_intrinsics=True
    )
    processor = ArUcoProcessor(camera=camera)

    # Load input image
    IN_IMG_PATH = "aruco_grid-in.png"
    OUT_IMG_PATH = "aruco_grid-out.png"
    in_img = cv2.imread(IN_IMG_PATH)

    # Process the image
    processed_img = processor.process_image(in_img)

    # Display and save the results
    cv2.imshow(
        "ArUco Marker Detection and Pose Estimation", 
        processed_img
    )
    cv2.imwrite(OUT_IMG_PATH, processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
