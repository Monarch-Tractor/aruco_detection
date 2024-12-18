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
from .detector import Detector


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


# def convert_rvec_tvec_to_pose(rvec, tvec):
#     """
#     This method converts the rotation and translation
#     vectors to a pose message.

#     Input:
#         - rvec: Rotation vector
#         - tvec: Translation vector

#     Output:
#         - Pose message
#     """
#     # Convert rvec to quaternion
#     rotation = R.from_rotvec(rvec.flatten())
#     quaternion = rotation.as_quat()

#     # Create the pose message
#     pose = Pose()
#     pose.position.x = tvec[0]
#     pose.position.y = tvec[1]
#     pose.position.z = tvec[2]
#     pose.orientation.x = quaternion[0]
#     pose.orientation.y = quaternion[1]
#     pose.orientation.z = quaternion[2]
#     pose.orientation.w = quaternion[3]

#     return pose


class Processor:
    """
    This class handles the detection of ArUco markers, 
    pose estimation, and camera pose estimation based 
    on the detected markers.
    """
    def __init__(self,
                 camera,
                 resize_image=False,
                 marker_length=0.05):
        """
        Initialize the processor with camera, marker length, 
        and ArUco dictionary type.

        Input:
            - camera: Camera object
            - resize_image: Flag to resize the input image
            - marker_length: Physical length of the marker 
                  in meters
        """
        self.camera = camera
        self.resize_image = resize_image
        self.marker_length = marker_length
        self.detector = Detector()

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
        # Draw axes for the marker.
        self.detector.draw_axes(
            rvec=rvec,
            tvec=tvec,
            intrinsic_matrix=self.camera.intrinsic_matrix,
            distortion_vector=self.camera.distortion_vector
        )

        # Draw the marker ID.
        x, y = int(corner[0][0]), int(corner[0][1])
        cv2.putText(
            self.resized_img,
            str(marker_id),
            (x, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 150, 250),  # (124, 214, 166),
            2,
            cv2.LINE_AA
        )

        return

    def process_image(self, img):
        """
        This method detect markers, estimates poses, and 
        overlays results on the image.

        Input:
            - img: Input image

        Output:
            - Processed image with detected markers and axes 
                  overlay
        """
        # Resize the image to 1280 x 720 (if enabled).
        if self.resize_image:
            self.detector.resized_img = cv2.resize(img, (1280, 720))
        else:
            self.detector.resized_img = img

        # Detect marker IDs and estimate poses.
        corners, ids, _ = self.detector.detect_markers()
        print("Detected markers:", ids)

        if ids is not None:
            # Draw detected markers.
            self.detector.draw_detected_markers(corners=corners)

            # Estimate poses of detected markers.
            rvecs, tvecs = self.detector.estimate_pose_multiple(
                corners=corners,
                marker_length=self.marker_length,
                intrinsic_matrix=self.camera.intrinsic_matrix,
                distortion_vector=self.camera.distortion_vector
            )

            # Post-process each marker image.
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

        # Set the camera pose estimate.
        self.relative_poses = {}
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            self.relative_poses[ids[i][0]] = (rvec, tvec)
        self.camera_pose = self.set_camera_pose_estimate()

        return self.detector.resized_img

    def get_relative_poses(self):
        """
        This method returns the relative poses of the 
        detected markers.
        """
        # Find the marker with the minimum relative
        # z-distance.
        min_id = min(
            self.relative_poses, 
            key=lambda x: self.relative_poses[x][1][2]
        )
        rel_pose = self.relative_poses[min_id]
        rel_pose = rvec_tvec_to_pose(
            rvec=rel_pose[0],
            tvec=rel_pose[1]
        )
        print(f"min_id: {min_id}")  # DEB
        print(f"rel_pose type: \n{type(rel_pose)}")  # DEB
        print(f"rel_pose: \n{rel_pose.position.z[0]}")  # DEB
        print("-" * 75)  # DEB
        return rel_pose
        # relative_poses = {}
        # relative_pose = None
        # for (id_, (rvec, tvec)) in self.relative_poses.items():
        #     print(f"rvec: \n{rvec}")  # DEB
        #     print(f"tvec: \n{tvec}")  # DEB
        #     print(f"z_dist [orig]: {tvec[2]}")  # DEB
        #     print("-" * 75)  # DEB
        #     relative_pose = rvec_tvec_to_pose(
        #         rvec=rvec,
        #         tvec=tvec
        #     )
        #     z_dist = relative_pose.position.z[0]
        #     relative_poses[id_] = z_dist
        #     print(f"z_dist [new]: {z_dist}")  # DEB
        #     print("-" * 75)  # DEB
        #     # print(f"id_: {id_}")  # DEB
        #     # print(f"relative_pose type: \n{type(relative_pose)}")  # DEB
        #     # print(f"relative_pose: \n{relative_pose.position.z[0]}")  # DEB
        #     # print("-" * 75)  # DEB
        #     break
        # return relative_pose

    def initialize_aruco_poses(self, c_rvec, c_tvec):
        """
        This method initializes the ArUco poses with 
        the given camera rotation and translation vectors.
        """
        for (id_, (rvec, tvec)) in self.relative_poses.items():
            rvec_global, tvec_global = transform_object_to_global(
                rvec_camera=c_rvec,
                tvec_camera=c_tvec,
                rvec_object_camera=rvec,
                tvec_object_camera=tvec
            )
            self.global_poses[id_] = (
                rvec_global, tvec_global
            )

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
        print(f"IDs: {ids}")  # DEB
        print("-" * 75)  # DEB
        print("-" * 75)  # DEB
        for id_ in ids:
            if id_ in self.relative_poses:
                rvec, tvec = self.relative_poses[id_]
                print(f"id_: {id_}")  # DEB
                print(f"rvec: \n{rvec}")  # DEB
                print(f"tvec: \n{tvec}")  # DEB
                print(f"rvec_global_object: \n{self.global_poses[id_]}")  # DEB
                print(f"tvec_global_object: \n{self.global_poses[id_]}")  # DEB
                print("-" * 75)  # DEB
                c_rvec, c_tvec = transform_camera_to_global(
                    rvec_object_camera=rvec,
                    tvec_object_camera=tvec,
                    rvec_object_global=self.global_poses[id_][0],
                    tvec_object_global=self.global_poses[id_][1]
                )
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
    processor = Processor(camera=camera)

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
