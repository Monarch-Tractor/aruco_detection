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

from .camera import Camera
from .detector import Detector
from .utils.ops import (
    encode_pose,
    transform_camera_to_global,
    transform_object_to_global
)


class ImageProcessor:
    """
    This class handles the detection of ArUco markers, 
    pose estimation, and camera pose estimation based 
    on the detected markers.
    """
    def __init__(self,
                 intrinsic_matrix,
                 distortion_vector,
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
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_vector = distortion_vector
        self.resize_image = resize_image
        self.marker_length = marker_length

        # AruCo marker detector
        self.detector = Detector()

    def postprocess_marker_image(self,
                                 rvec,
                                 tvec,
                                 corner,
                                 marker_id):
        """
        This method post-process a single marker by drawing 
        its axes and overlaying its ID.

        Input:
            - rvec: Rotation vector
            - tvec: Translation vector
            - corner: Marker corner (used for text placement)
            - marker_id: ID of the marker
            - img: Resized image to overlay results

        Output:
            - None (modifies the image in place)
        """
        # Draw axes for the marker
        self.detector.draw_axes(
            rvec=rvec,
            tvec=tvec,
            intrinsic_matrix=self.intrinsic_matrix,
            distortion_vector=self.distortion_vector
        )

        # Draw the marker ID below the detected marker
        x, y = int(corner[0][0]), int(corner[0][1])  # Top-left corner
        cv2.putText(
            self.detector.resized_img,
            str(marker_id),
            (x, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 150, 250),
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
            - timestamp: Timestamp of the image

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
                intrinsic_matrix=self.intrinsic_matrix,
                distortion_vector=self.distortion_vector
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

        return (
            self.detector.resized_img,
            ids,
            rvecs,
            tvecs
        )


class PoseProcessor:
    """
    This class estimates the global pose as well as the
    relative pose of the camera with respect to a detected
    ArUco marker.
    """
    def __init__(self):
        """
        Initialize the processor with camera, marker length, 
        and ArUco dictionary type.
        """

        # Camera pose initialization
        self.camera_pose = encode_pose(
            rvec=np.zeros(3, dtype=np.float32),
            tvec=np.zeros(3, dtype=np.float32)
        )

        # Relative and global poses
        self.relative_poses = {}
        self.global_poses = {}

    def process_single_pose(self,
                            marker_id,
                            rvec,
                            tvec):
            """
            This method is a helper function to process a 
            single pose.
            """
            return marker_id[0], (rvec, tvec)

    def process_pose(self, ids, rvecs, tvecs):
        """
        This method processes the estimated pose of the
        camera with respect to the detected ArUco markers
        using parallelization and eliminating explicit for
        loops.

        Input:
            - ids: Array of marker IDs
            - rvecs: Array of rotation vectors
            - tvecs: Array of translation vectors
        """
        # Process each pose in parallel.
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                self.process_single_pose,
                ids,
                rvecs,
                tvecs
            ))
        # Convert results to a dictionary
        self.relative_poses = dict(results)
        # Set the camera pose estimate
        self.camera_pose = self.set_camera_pose_estimate()

        global_pose = self.get_camera_pose_estimate()
        relative_pose = self.get_relative_pose()
        return global_pose, relative_pose

    def get_relative_pose(self):
        """
        This method returns the relative pose of the 
        detected markers which is closest to the camera.
        """
        # Find the marker with the minimum relative
        # z-distance.
        min_id = min(
            self.relative_poses,
            key=lambda x: self.relative_poses[x][1][2]
        )
        # Get the pose of the marker with the minimum
        # relative z-distance.
        target_relative_pose = self.relative_poses[min_id]
        # Encode the pose as a Pose message.
        target_relative_pose = encode_pose(
            rvec=target_relative_pose[0],
            tvec=target_relative_pose[1]
        )
        return target_relative_pose

    def initialize_aruco_poses(self, c_rvec, c_tvec):
        """
        This method initializes the ArUco poses with 
        the given camera rotation and translation vectors.
        """
        for (id_, (rvec, tvec)) in self.relative_poses.items():
            rvec_global, tvec_global = \
                transform_object_to_global(
                    rvec_camera=c_rvec,
                    tvec_camera=c_tvec,
                    rvec_object_camera=rvec,
                    tvec_object_camera=tvec
                )
            self.global_poses[id_] = (
                rvec_global, tvec_global
            )
        return

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
                c_rvec, c_tvec = \
                    transform_camera_to_global(
                        rvec_object_camera=rvec,
                        tvec_object_camera=tvec,
                        rvec_object_global=self.global_poses[id_][0],
                        tvec_object_global=self.global_poses[id_][1]
                    )
                self.camera_pose = encode_pose(
                    rvec=c_rvec,
                    tvec=c_tvec
                )
        return self.camera_pose

    def get_camera_pose_estimate(self):
        """
        This method returns the camera pose estimate.
        """
        return self.camera_pose


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

        # Image processor
        self.image_processor = ImageProcessor(
            intrinsic_matrix=self.camera.intrinsic_matrix,
            distortion_vector=self.camera.distortion_vector,
            resize_image=resize_image,
            marker_length=marker_length
        )

        # Pose processor
        self.pose_processor = PoseProcessor()

    def process(self, img):
        """
        This method detect markers, estimates poses
        (in different coordinate frames), and publishes
        the results as ROS messages

        Input:
            - img: Input image
        """
        # Process the image.
        rgb_out, ids, rvecs, tvecs = \
            self.image_processor.process_image(img=img)

        # Process the pose.
        global_pose, reltaive_pose = \
            self.pose_processor.process_pose(
                ids=ids,
                rvecs=rvecs,
                tvecs=tvecs
            )

        return rgb_out, global_pose, reltaive_pose


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
    img_processor = ImageProcessor(
        intrinsic_matrix=camera.intrinsic_matrix,
        distortion_vector=camera.distortion_vector,
    )

    # Load input image
    IN_IMG_PATH = "aruco_grid-in.png"
    OUT_IMG_PATH = "aruco_grid-out.png"
    in_img = cv2.imread(IN_IMG_PATH)

    # Process the image
    processed_img, _, _, _ = \
        img_processor.process_image(in_img)

    # Display and save the results
    cv2.imshow(
        "ArUco Marker Detection and Pose Estimation", 
        processed_img
    )
    cv2.imwrite(OUT_IMG_PATH, processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
