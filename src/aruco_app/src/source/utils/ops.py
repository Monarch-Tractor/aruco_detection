"""
This script contains helper methods used across the project.
"""


import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import Pose

from scipy.spatial.transform import Rotation as R


# Helper methods for encoding data into ROS messages.
def encode_image(img, timestamp):
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

def encode_pose(rvec, tvec):
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

def encode_odometry(pose,
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


# Helper methods for decoding incoming ROS messages.
def decode_image(msg):
    """
    This method decodes a sensor_msgs/CompressedImage
    message into a numpy array.
    """
    # Decode the compressed image message.
    img = np.frombuffer(
        buffer=msg.data,
        dtype=np.uint8
    )

    img = cv2.imdecode(
        buf=img,
        flags=cv2.IMREAD_COLOR
    )
    return img


def decode_pose(msg):
    """
    This method decodes a geometry_msgs/Pose message
    into a rotation vector and translation vector.
    """
    rvec, tvec = None, None

    if msg is not None:
        # Extract the quaternion from the pose message.
        quaternion = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

        # Convert the quaternion to a rotation vector.
        rotation = R.from_quat(quaternion)
        rvec = rotation.as_rotvec().reshape(3, 1)

        # Extract the translation vector from the pose message.
        tvec = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]).reshape(3, 1)

    return rvec, tvec


# Helper methods to perform pose transformations between
# different coordinate frames.
def transform_camera_to_global(rvec_object_camera,
                               tvec_object_camera,
                               rvec_object_global,
                               tvec_object_global):
    """
    This method transforms the camera's pose (rotation and 
    translation) from the object's frame to the global frame,
    given the pose of the object in both the camera and global 
    frames.

    Input:
        - rvec_object_camera: Orientation of the object relative 
              to the camera's coordinate frame
        - tvec_object_camera: Position of the object relative
              to the camera's coordinate frame
        - rvec_object_global: Orientation of the object relative
              to the global coordinate frame
        - tvec_object_global : Position of the object relative
              to the global coordinate frame

    Output:
        - rvec_camera: Orientation of the camera relative to the
              global coordinate frame
        - t_camera: Position of the camera relative to the 
              global coordinate frame
    """
    # Convert rvecs to rotation matrices.
    r_object_camera, _ = cv2.Rodrigues(rvec_object_camera)
    r_object_global, _ = cv2.Rodrigues(rvec_object_global)

    # Compute camera rotation.
    r_camera = r_object_global.T @ r_object_camera

    # Compute camera translation.
    t_camera = r_object_global.T @ (
        tvec_object_camera - tvec_object_global
    )

    # Convert rotation matrix back to rvec.
    rvec_camera, _ = cv2.Rodrigues(r_camera)

    return rvec_camera, t_camera

def transform_object_to_global(rvec_camera,
                               tvec_camera,
                               rvec_object_camera,
                               tvec_object_camera):
    """
    This method transforms the pose (rotation and translation)
    of an object from the camera's frame to the global frame,
    given the pose of the camera in the global frame.

    Input:
        - rvec_camera: Orientation of the camera in the global
              coordinate frame
        - tvec_camera: Position of the camera in the global 
              coordinate frame
        - rvec_object_camera: Orientation of the object in the 
              camera's coordinate frame.
        - tvec_object_camera: Position of the object in the
              camera's coordinate frame

    Output:
        - rvec_object_global: Orientation of the object in the
              global coordinate frame
        - t_object_global: Position of the object in the
              global coordinate frame
    """
    # Convert rvecs to rotation matrices.
    r_camera, _ = cv2.Rodrigues(rvec_camera)
    r_object_camera, _ = cv2.Rodrigues(rvec_object_camera)

    # Compute global rotation.
    r_object_global = r_camera @ r_object_camera

    # Compute global translation.
    t_object_global = r_camera @ (tvec_object_camera + tvec_camera)

    # Convert rotation matrix back to rvec.
    rvec_object_global, _ = cv2.Rodrigues(r_object_global)

    return rvec_object_global, t_object_global

