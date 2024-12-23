#!/usr/bin/env python3


"""
This script contains the class definition for the 
ArUco application. The ArUcoApp class reads input images
from a ROS topic, processes the images using the ArUco
marker detection and pose estimation algorithm, and
publishes the output as ROS messages.
"""


import yaml

import numpy as np

import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R

from source.camera import Camera
from source.processor import Processor
from source.io_handler import (
    Input, Output
)

from source.utils.ops import (
    encode_image,
    encode_odometry,
    decode_image,
    decode_pose
)


class ArUcoApp:
    """
    This class is the ROS handler which receives incoming
    messages from different topics, processes the messages
    and publishes the output to different topics.
    """
    def __init__(self):
        self.name = "aruco_app_node"

        # Initialize ROS Node.
        rospy.init_node(
            name=self.name,
            anonymous=True
        )

        # Load config file.
        config_path = rospy.get_param(
            "~config"
        )
        with open(config_path, "r") as config_file:
            self.config = yaml.load(
                stream=config_file,
                Loader=yaml.FullLoader
            )

        # Setup the node parameters.
        # Loop rate
        self.rate_hz = self.config["app_cfg"]["loop_rate"]
        # Camera name
        self.camera_name = self.config["app_cfg"]["camera_name"]
        # Frame IDs
        self.camera_frame_id = self.config["app_cfg"]["camera_frame_id"]
        self.baselink_frame_id = self.config["app_cfg"]["baselink_frame_id"]
        # Subscriber topics
        self.camera_info_topic = self.config["app_cfg"]["msg_cfg"]\
            ["topics"]["subscribe"]["camera_info_topic"]
        self.rgb_in_topic = self.config["app_cfg"]["msg_cfg"]\
            ["topics"]["subscribe"]["rgb_topic"]
        self.service_topic = self.config["app_cfg"]["msg_cfg"]\
            ["topics"]["subscribe"]["service_topic"]
        self.localization_topic = self.config["app_cfg"]["msg_cfg"]\
            ["topics"]["subscribe"]["localization_topic"]
        self.tf_static_topic = self.config["app_cfg"]["msg_cfg"]\
            ["topics"]["subscribe"]["tf_static_topic"]
        # Publisher topics
        self.rgb_out_topic = self.config["app_cfg"]["msg_cfg"]\
            ["topics"]["publish"]["rgb_topic"]
        self.relative_pose_topic = self.config["app_cfg"]["msg_cfg"]\
            ["topics"]["publish"]["relative_pose_topic"]
        self.global_pose_topic = self.config["app_cfg"]["msg_cfg"]\
            ["topics"]["publish"]["global_pose_topic"]
        
        # Input data attribute
        self.a_in = Input()

        # Output data attribute
        self.a_out = Output()

        # Define subscribers and publishers.
        self.set_subscribers()
        self.set_publishers()

        # Initialize the tf buffer.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer
        )
        self.lookup_static_transform()

        self.aruco_poses = {}

        # TODO: Add methods for estimating the camera pose from detected markers

        # Get camera instance.
        self.camera = Camera(
            camera_name=self.camera_name,
            camera_info_topic=self.camera_info_topic
        )

        # Processor instance
        self.processor = Processor(
            camera=self.camera
        )

    def set_subscribers(self):
        """
        This method defines the different subscribers.
        """
        # RGB data subscriber
        self.rgb_sub = rospy.Subscriber(
            name=self.rgb_in_topic,
            data_class=CompressedImage,
            callback=self.rgb_topic_callback
        )
        # Service call subscriber
        self.init_aruco_pose_service = rospy.Subscriber(
            name=self.service_topic,
            data_class=Bool,
            callback=self.initialize_aruco_pose_callback
        )
        # Localization data subscriber
        self.localization_sub = rospy.Subscriber(
            name=self.localization_topic,
            data_class=Odometry,
            callback=self.localization_callback
        )
        return

    def set_publishers(self):
        """
        This method defines the different publishers.
        """
        # Ouput RGB data publisher
        self.rgb_pub= rospy.Publisher(
            name=self.rgb_out_topic,
            data_class=CompressedImage,
            queue_size=10
        )
        # Relative pose publisher
        self.relative_pose_pub = rospy.Publisher(
            name=self.relative_pose_topic,
            data_class=Odometry,
            queue_size=10
        )
        # Global pose publisher
        self.global_pose_pub = rospy.Publisher(
            name=self.global_pose_topic,
            data_class=Odometry,
            queue_size=10
        )
        return

    def initialize_aruco_pose_callback(self, msg):
        """
        This method initializes the ArUco pose.
        """
        if msg.data:
            rospy.loginfo("Initializing ArUco pose.")
            rvec, tvec = decode_pose(
                msg=self.a_in.camera_pose
            )
            self.processor.initialize_aruco_poses(
                c_rvec=rvec,
                c_tvec=tvec
            )
            self.aruco_poses = \
                self.processor.get_global_poses()
        return

    def lookup_static_transform(self):
        """
        This method looks up the static transform from
        the baselink frame to the camera frame. This
        transform is used to estimate the camera pose
        from the odometry data.
        """
        try:
            self.a_in.baselink_to_camera = \
                self.tf_buffer.lookup_transform(
                    target_frame=self.baselink_frame_id,
                    source_frame=self.camera_frame_id,
                    time=rospy.Time(0),
                    timeout=rospy.Duration(5.0)
                )
            rospy.loginfo(
                "Static transform from /baselink " + \
                "to /camera obtained."
            )
        except tf2_ros.LookupException as e:
            rospy.logerr(
                f"Failed to lookup static transform: {e}"
            )
        return

    def localization_callback(self, msg):
        """
        This method is the callback function for the
        odometry topic. It is called whenever a new
        odometry message is received, and the received
        message is used to update the camera pose.
        """
        self.a_in.baselink_pose = msg.pose
        self.a_in.camera_pose = \
            tf2_geometry_msgs.do_transform_pose(
                pose=self.a_in.baselink_pose,
                transform=self.a_in.baselink_to_camera
            )
        return


    def rgb_topic_callback(self, rgb_msg):
        """
        This method decodes an incoming compressed image
        message and stores the decoded image in the input
        data attribute.
        """
        # Callback function for RGB topic.
        rospy.loginfo("Received RGB data from RGB topic.")

        # Decode and store the RGB data in the input
        # data attribute.
        rgb_in = decode_image(msg=rgb_msg)
        self.a_in.update(rgb_data=rgb_in)
        return

    def encode_output(self,
                      rgb_data,
                      global_pose,
                      relative_pose):
        """
        This method encodes the output data into ROS
        messages.
        """
        # Get the current timestamp.
        timestamp = rospy.Time.now()

        # Initialize the encoded output attributes.
        encoded_rgb_data = None
        encoded_global_pose = None
        encoded_relative_pose = None

        # Encode the output image.
        if rgb_data is not None:
            encoded_rgb_data = encode_image(
                img=rgb_data,
                timestamp=timestamp
            )
        # Encode the global pose.
        if global_pose is not None:
            encoded_global_pose = encode_odometry(
                pose=global_pose,
                frame_id="map",
                child_frame_id="zed_front_left_img",
                timestamp=timestamp
            )
        # Encode the relative pose.
        if relative_pose is not None:
            encoded_relative_pose = encode_odometry(
                pose=relative_pose,
                frame_id="map",
                child_frame_id="zed_front_left_img",
                timestamp=timestamp
            )

        return (
            encoded_rgb_data,
            encoded_global_pose,
            encoded_relative_pose
        )

    def run(self):
        """
        This method runs the main loop of the node
        at the specified rate.
        """
        # Adjust loop rate.
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            # Publish the visualizer message to the output
            # topics.
            self.process()
            self.reset()
            rate.sleep()

        return

    def reset(self):
        """
        This method resets the input data attributes.
        """
        self.a_in.reset(reset_rgb_data=True)
        return

    def process(self):
        """
        This method reads the input data, processes it
        using the ArUco marker detection and pose estimation
        algorithm, and publishes the output.
        """
        if self.a_in.rgb_data is not None:
            # Process the input image and generate the
            # output image with detected ArUco markers'
            # Id s and poses.
            rgb_out, \
            global_pose, \
            relative_pose = self.processor.process(
                img=self.a_in.rgb_data
            )

            # Update the output attribute.
            self.a_out.update(rgb_data=rgb_out)

            # Encode the processor's outputs.
            encoded_rgb_out, \
            encoded_global_pose, \
            encoded_relative_pose = self.encode_output(
                rgb_data=rgb_out,
                global_pose=global_pose,
                relative_pose=relative_pose
            )

            # Publish the encoded output messages.
            if encoded_rgb_out is not None:
                self.rgb_pub.publish(
                    encoded_rgb_out
                )
            if encoded_global_pose is not None:
                self.global_pose_pub.publish(
                    encoded_global_pose
                )
            if encoded_relative_pose is not None:
                self.relative_pose_pub.publish(
                    encoded_relative_pose
                )

        return


if __name__ == '__main__':
    # Initialize the ArUco app ROS node.
    aruco_reader = ArUcoApp()

    # Run the ArUco reader node.
    aruco_reader.run()
