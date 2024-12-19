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

import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

import tf2_ros
import tf2_geometry_msgs

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose

from .camera import Camera
from .detector import Detector


# def rvec_tvec_to_pose(rvec, tvec):
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


class ImageProcessor:
    """
    This class handles the detection of ArUco markers, 
    pose estimation, and camera pose estimation based 
    on the detected markers.
    """
    def __init__(self,
                 camera,
                 resize_image=False,
                 marker_length=0.05,
                 rgb_topic=None):
        """
        Initialize the processor with camera, marker length, 
        and ArUco dictionary type.

        Input:
            - camera: Camera object
            - resize_image: Flag to resize the input image
            - marker_length: Physical length of the marker 
                  in meters
            - rgb_topic: RGB image topic
        """
        self.camera = camera
        self.resize_image = resize_image
        self.marker_length = marker_length
        self.rgb_topic = rgb_topic

        # AruCo marker detector
        self.detector = Detector()

        # Publishers
        self.set_publishers()

    def set_publishers(self):
        """
        This method defines the different publishers.
        """
        if self.rgb_topic is not None:
            self.rgb_pub= rospy.Publisher(
                name=self.rgb_topic,
                data_class=CompressedImage,
                queue_size=10
            )
        return

    def encode_image(self, img, timestamp):
        """
        This method encodes a numpy array of an image into a 
        sensor_msgs/CompressedImage message.
        """
        msg = CompressedImage()
        msg.header = Header(
            frame_id="map",
            stamp=timestamp,
        )
        _, compressed_img = cv2.imencode(".jpg", img)
        msg.format = "jpeg"
        msg.data = np.asarray(compressed_img).tobytes()
        return msg

    def postprocess_output(self,
                           rgb_data,
                           timestamp):
        """
        This method post-processes the output of the ArUco
        marker detection and pose estimation algorithm.
        """
        # Encode output RGB data.
        encoded_rgb_data = None
        if rgb_data is not None:
            encoded_rgb_data = self.encode_image(
                img=rgb_data,
                timestamp=timestamp
            )

        return encoded_rgb_data

    def publish_output(self,
                       rgb_msg=None):
        """
        This method publishes the output of the
        ArUco marker detection and pose estimation
        algorithm.
        """
        # Publish ID and pose of the detected ArUco
        # markers as a compressed image.
        if rgb_msg is not None:
            self.rgb_pub.publish(
                rgb_msg
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

    def process_image(self, img, timestamp=None):
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

        if ((self.rgb_topic is not None) and\
            (self.detector.resized_img is not None)):
            # Encode the image, relative pose, and global pose
            # to publish as ROS messages.
            encoded_rgb_data = self.postprocess_output(
                rgb_data=self.detector.resized_img,
                timestamp=timestamp
            )
            # Publish the encoded output as ROS messages.
            self.publish_output(rgb_msg=encoded_rgb_data)

        return (
            self.detector.resized_img,
            ids, rvecs, tvecs
        )


class PoseProcessor:
    """
    This class estimates the global pose as well as the
    relative pose of the camera with respect to a detected
    ArUco marker.
    """
    def __init__(self,
                 camera,
                 relative_pose_topic=None,
                 global_pose_topic=None):
        """
        Initialize the processor with camera, marker length, 
        and ArUco dictionary type.

        Input:
            - camera: Camera object
            - relative_pose_topic: Relative pose topic
            - global_pose_topic: Global pose topic
        """
        self.camera = camera
        self.relative_pose_topic = relative_pose_topic
        self.global_pose_topic = global_pose_topic

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

        # Publishers
        self.set_publishers()

    def set_publishers(self):
        """
        This method defines the different publishers.
        """
        if self.relative_pose_topic is not None:
            self.relative_pose_pub = rospy.Publisher(
                name=self.relative_pose_topic,
                data_class=Odometry,
                queue_size=10
            )
        if self.global_pose_topic is not None:
            self.global_pose_pub = rospy.Publisher(
                name=self.global_pose_topic,
                data_class=Odometry,
                queue_size=10
            )
        return

    def transform_object_to_global(self,
                                   rvec_camera,
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
        t_object_global = \
            R_camera @ tvec_object_camera + tvec_camera

        # Convert rotation matrix back to rvec
        rvec_object_global, _ = cv2.Rodrigues(R_object_global)

        return rvec_object_global, t_object_global


    def transform_camera_to_global(self,
                                   rvec_object_camera,
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

    def encode_pose(self, rvec, tvec):
        """
        This method encodes the camera pose estimate into
        a geometry_msgs/Pose message.
        """
        # Convert rvec to quaternion.
        rotation = R.from_rotvec(rvec.flatten())
        quaternion = rotation.as_quat()

        # Create the pose message.
        pose = Pose()
        pose.position.x = tvec[0]
        pose.position.y = tvec[1]
        pose.position.z = tvec[2]
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose

    def encode_odometry(self,
                        pose,
                        frame_id,
                        child_frame_id,
                        timestamp):
        """
        This method encodes a pose into a nav_msgs/Odometry
        message.
        """
        msg = Odometry()
        msg.header = Header(
            frame_id=frame_id,
            stamp=timestamp
        )
        msg.child_frame_id = child_frame_id
        msg.pose.pose = pose
        return msg

    def postprocess_output(self,
                           relative_pose,
                           global_pose,
                           timestamp):
        """
        This method post-processes the estimated poses of
        the camera in different coordinate frames.
        """
        # Encode relative pose odometry.
        encoded_relative_pose = None
        if relative_pose is not None:
            encoded_relative_pose = self.encode_odometry(
                pose=relative_pose,
                frame_id="map",
                child_frame_id="zed_front_left_msg",
                timestamp=timestamp
            )

        # Encode global pose odometry.
        encoded_global_pose = None
        if global_pose is not None:
            encoded_global_pose = self.encode_odometry(
                pose=global_pose,
                frame_id="map",
                child_frame_id="zed_front_left_msg",
                timestamp=timestamp
            )

        return (
            encoded_relative_pose,
            encoded_global_pose
        )

    def publish_output(self,
                       relative_pose_msg=None,
                       global_pose_msg=None):
        """
        This method publishes the estimated output 
        camera poses in different coordinate frames.
        """
        # Publish the relative pose of the camera
        # with respect to the detected ArUco markers.
        if relative_pose_msg is not None:
            self.relative_pose_pub.publish(
                relative_pose_msg
            )
        # Publish the global pose of the camera.
        if global_pose_msg is not None:
            self.global_pose_pub.publish(
                global_pose_msg
            )
        return

    def process_pose(self,
                     ids,
                     rvecs,
                     tvecs,
                     timestamp=None):
        """
        This method processes the estimated pose of the
        camera with respect to the detected ArUco markers.

        Input:
            - pose: Estimated pose of the camera with respect
                  to the detected ArUco markers
            - timestamp: Timestamp of the pose estimation
        """
        # Set the camera pose estimate.
        self.relative_poses = {}
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            self.relative_poses[ids[i][0]] = (rvec, tvec)
        self.camera_pose = self.set_camera_pose_estimate()

        if ((self.relative_pose_topic is not None) and \
            (self.global_pose_topic is not None)):
            # Encode the relative and global pose of the
            # camera to publish as ROS messages.
            relative_pose = self.get_relative_pose()
            global_pose = self.get_camera_pose_estimate()
            encoded_relative_pose, \
            encoded_global_pose = self.postprocess_output(
                relative_pose=relative_pose,
                global_pose=global_pose,
                timestamp=timestamp
            )
            # Publish the encoded output as ROS messages.
            self.publish_output(
                relative_pose_msg=encoded_relative_pose,
                global_pose_msg=encoded_global_pose
            )

        return

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
        target_relative_pose = self.encode_pose(
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
                self.transform_object_to_global(
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
                c_rvec, c_tvec = \
                    self.transform_camera_to_global(
                        rvec_object_camera=rvec,
                        tvec_object_camera=tvec,
                        rvec_object_global=self.global_poses[id_][0],
                        tvec_object_global=self.global_poses[id_][1]
                    )
                self.camera_pose = self.encode_pose(
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
                 marker_length=0.05,
                 rgb_topic=None,
                 relative_pose_topic=None,
                 global_pose_topic=None):
        """
        Initialize the processor with camera, marker length, 
        and ArUco dictionary type.

        Input:
            - camera: Camera object
            - resize_image: Flag to resize the input image
            - marker_length: Physical length of the marker 
                  in meters
            - rgb_topic: RGB image topic
            - relative_pose_topic: Relative pose topic
            - global_pose_topic: Global pose topic
        """
        self.camera = camera
        self.resize_image = resize_image
        self.marker_length = marker_length
        self.rgb_topic = rgb_topic
        self.relative_pose_topic = relative_pose_topic
        self.global_pose_topic = global_pose_topic

        # Image processor
        self.image_processor = ImageProcessor(
            camera=camera,
            resize_image=resize_image,
            marker_length=marker_length,
            rgb_topic=rgb_topic
        )

        # Pose processor
        self.pose_processor = PoseProcessor(
            camera=camera,
            relative_pose_topic=relative_pose_topic,
            global_pose_topic=global_pose_topic
        )

    def process(self, img):
        """
        This method detect markers, estimates poses
        (in different coordinate frames), and publishes
        the results as ROS messages

        Input:
            - img: Input image
        """
        # Get timestamp.
        timestamp = rospy.Time.now()

        # Process the image.
        _, ids, rvecs, tvecs = \
            self.image_processor.process_image(
                img=img,
                timestamp=timestamp
            )

        # Process the pose.
        self.pose_processor.process_pose(
            ids=ids,
            rvecs=rvecs,
            tvecs=tvecs,
            timestamp=timestamp
        )

        return


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
    img_processor = ImageProcessor(camera=camera)

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
