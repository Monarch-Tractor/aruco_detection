"""
This script contains the class definitions for the camera.
The Camera class is used to represent the camera with its 
intrinsic and distortion parameters.
"""


import numpy as np

import rospy
from sensor_msgs.msg import CameraInfo


class Camera:
    """
    This class represents the camera with its 
    intrinsic and distortion parameters.
    """
    def __init__(self,
                 camera_name,
                 camera_info_topic,
                 use_default_intrinsics=False):
        """
        This class initializes the camera with intrinsic 
        matrix and distortion coefficients.

        Input:
            - camera_name: Name of the camera
            - intrinsic_matrix: Intrinsic camera matrix
            - distortion_coefficients: Distortion coefficients
        """
        # Camera name and ROS topic
        self.camera_name = camera_name
        self.camera_info_topic = camera_info_topic

        # Camera information
        # Camera dimensions
        self.camera_width = None
        self.camera_height = None
        # Camera parameters
        self.intrinsic_matrix = None
        self.distortion_vector = None

        # Flag to use default intrinsic parameters
        self.use_default_intrinsics = use_default_intrinsics

        # Flag to check if the camera information is
        # received.
        self.information_received = False
        # Timeout duration
        self.timeout = 5.0

        # Camera information topic subscriber
        self.camera_info_sub = rospy.Subscriber(
            name=self.camera_info_topic,
            data_class=CameraInfo,
            callback=self.camera_info_topic_callback
        )

        # Read camera information.
        self.wait_for_information()
        if not self.information_received:
            msg = "Using default camera parameters because " + \
               "no camera parameter information was received."
            rospy.loginfo(msg)
            self.use_default_intrinsics = True
            self.use_default_camera_parameters()

        # Print camera information.
        print(f"intrinsic_matrix shape: {self.intrinsic_matrix.shape}")
        print(f"intrinsic_matrix: \n{self.intrinsic_matrix}")
        print("-" * 75)
        print(f"distortion_vector shape: {self.distortion_vector.shape}")
        print(f"distortion_vector: \n{self.distortion_vector}")
        print("-" * 75)

    def use_default_intrinsics_type1(self):
        """
        This method returns default camera intrinsic parameters 
        (type 1).
        """
        # Intrinsic matrix
        self.intrinsic_matrix = np.array([
            [1093.27,     0.0,  965.0],
            [    0.0, 1093.27,  569.0],
            [    0.0,     0.0,    1.0]
        ], dtype=np.float32)
        # Distortion coefficients
        self.distortion_vector = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32
        )
        return

    def camera_info_topic_callback(self, msg):
        """
        This callback method decodes the camera information
        from the cameradistortion_vector information topic if not already
        received.
        """
        if not self.information_received:
            rospy.loginfo("Received camera information.")
            # Set flag to True.
            self.information_received = True
            # Extract camera information.
            # Extract camera dimensions.
            self.camera_width = msg.width
            self.camera_height = msg.height

            # Extract camera parameters.
            self.intrinsic_matrix = np.array(
                msg.K,
                dtype=np.float32
            ).reshape((3, 3))
            self.distortion_vector = np.array(
                msg.D,
                dtype=np.float32
            )

            # Use default values for camera intrinsic and
            # extrinsic parameters if flags are set to true.
            self.use_default_camera_parameters()

    def use_default_camera_parameters(self):
        """
        This method returns default camera intrinsic and 
        extrinsic parameters.
        """
        if self.use_default_intrinsics:
            self.use_default_intrinsics_type1()

        return

    def wait_for_information(self):
        """
        This method waits for the camera information to be
        received.
        """
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time) < rospy.Duration(self.timeout):
            if self.information_received:
                rospy.loginfo("Camera information received within the timeout.")
                return
            rospy.sleep(0.1)  # Sleep to avoid busy-waiting
        rospy.logwarn("Timeout reached without receiving camera information.")
        # Unsubscribe from the topic
        self.camera_info_sub.unregister()
        rospy.loginfo("Unsubscribed from the topic.")
        return
    