"""
This script contains the class definition for the 
ArUco application. The App class reads input images
from a ROS topic, processes the images using the ArUco
marker detection and pose estimation algorithm, and
publishes the output as ROS messages.
"""

import numpy as np
import cv2

import rospy
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R

from source.camera import Camera
from source.processor import Processor
from source.io_handler import Input
from source.publishers import IDPosePublisher


def pose_to_rvec_tvec(pose):
    """
    This method converts a geometry_msgs/Pose message
    to a rotation vector and translation vector.
    """
    # Extract translation vector (tvec).
    tvec = np.array([
        pose.position.x,
        pose.position.y,
        pose.position.z
    ], dtype=np.float64)

    # Extract quaternion.
    quaternion = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ]

    # Convert quaternion to rotation vector (rvec).
    rotation = R.from_quat(quaternion)
    # Convert the rotation to a rotation vector.
    rvec = rotation.as_rotvec()

    return (
        rvec.reshape(3, 1),
        tvec
    )


class App:
    """
    This class defines the ArUco application that reads
    input images from a ROS topic, processes the images
    using the ArUco marker detection and pose estimation
    algorithm, and publishes the output as ROS messages.
    """
    def __init__(self,
                 camera_name,
                 camera_info_topic,
                 rgb_in_topic,
                 rgb_out_topic,
                 rate_hz,
                 service_topic,
                 localization_topic,
                 tf_static_topic):
        self.camera_name = camera_name
        self.camera_info_topic = camera_info_topic

        self.rgb_in_topic = rgb_in_topic
        self.rgb_out_topic = rgb_out_topic

        self.rate_hz = rate_hz

        self.service_topic = service_topic
        self.localization_topic = localization_topic
        self.tf_static_topic = tf_static_topic

        self.relative_pose_topic = "/aruco_reader/relative_pose"
        self.global_pose_topic = "/aruco_reader/global_pose"

        self.camera_frame = 'zed_front_left_img'
        self.baselink_frame = 'base_link'

        # Initialize the ROS node.
        rospy.init_node(
            name="aruco_marker_reader_node",
            anonymous=True
        )

        # Define subscribers.
        self.set_subscribers()
        self.set_publishers()

        # # Define the service call.
        # self.set_services()

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

        # Get ArUco processor instance.
        self.aruco_processor = Processor(
            camera=self.camera
        )

        # Publishers for the topics.
        self.aruco_id_pose_publisher = IDPosePublisher(
            camera_name=self.camera_name,
            rgb_out_topic=self.rgb_out_topic,
            relative_pose_topic=self.relative_pose_topic,
            global_pose_topic=self.global_pose_topic
        )

        # Input data attribute.
        self.a_in = Input()

        # Output data attribute.
        self.rgb_out = None

    def set_publishers(self):
        """
        This method defines the different publishers.
        """
        self.relative_pose_pub = rospy.Publisher(
            name=self.relative_pose_topic,
            data_class=Odometry,
            queue_size=10
        )

        self.global_pose_pub = rospy.Publisher(
            name=self.global_pose_topic,
            data_class=Odometry,
            queue_size=10
        )

        return

    def set_subscribers(self):
        """
        This method defines the different subscribers.
        """
        self.rgb_in_sub = rospy.Subscriber(
            name=self.rgb_in_topic,
            data_class=CompressedImage,
            callback=self.rgb_topic_callback
        )

        self.initialize_aruco_pose_service = rospy.Subscriber(
            name=self.service_topic,
            data_class=Bool,
            callback=self.initialize_aruco_pose_callback
        )

        self.localization_sub = rospy.Subscriber(
            name=self.localization_topic,
            data_class=Odometry,
            callback=self.localization_callback
        )
        return

    def initialize_aruco_pose_callback(self, msg):
        """
        This method initializes the ArUco pose.
        """
        if msg.data:
            rospy.loginfo("Initializing ArUco pose.")
            rvec, tvec = pose_to_rvec_tvec(
                pose=self.camera_pose
            )
            self.aruco_processor.initialize_aruco_poses(
                c_rvec=rvec,
                c_tvec=tvec
            )
            self.aruco_poses = \
                self.aruco_processor.get_global_poses()
        return

    def decode_image(self, msg):
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

    def lookup_static_transform(self):
        """
        This method looks up the static transform from
        the camera frame to the baselink frame. This
        transform is used to estimate the camera pose
        from the odometry data.
        """
        try:
            self.baselink_to_camera = \
                self.tf_buffer.lookup_transform(
                    target_frame=self.baselink_frame,
                    source_frame=self.camera_frame,
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
        self.baselink_pose = msg.pose
        self.camera_pose = \
            tf2_geometry_msgs.do_transform_pose(
                pose=self.baselink_pose,
                transform=self.baselink_to_camera
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
        rgb_in = self.decode_image(msg=rgb_msg)
        self.a_in.update(rgb_data=rgb_in)
        return

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
        self.a_in.reset()
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
            rgb_out = self.aruco_processor.process_image(
                img=self.a_in.rgb_data
            )
            self.a_out.update(rgb_data=rgb_out)
            rgb_out_encoded = \
                self.aruco_id_pose_publisher.encode_image(
                    img=rgb_out,
                    timestamp=rospy.Time.now()
                )
            self.aruco_id_pose_publisher.publish_output(
                rgb_out_msg=rgb_out_encoded
            )

            # Get the global pose of the camera, and publish
            # the global pose.
            global_pose = self.aruco_processor.get_camera_pose_estimate()
            if global_pose is not None:
                global_pose_encoded = \
                    self.aruco_id_pose_publisher.encode_pose(
                        pose=global_pose,
                        frame_id="map",
                        child_frame_id="zed_front_left_img",
                        timestamp=rospy.Time.now()
                    )
                self.aruco_id_pose_publisher.publish_output(
                    global_pose_msg=global_pose_encoded
                )

            # Get the relative pose of the camera with respect
            # to the detected ArUco markers, and publish the
            # relative pose.
            relative_pose = self.aruco_processor.get_relative_poses()
            if relative_pose is not None:
                relative_pose_encoded = \
                    self.aruco_id_pose_publisher.encode_pose(
                        pose=relative_pose,
                        frame_id="map",
                        child_frame_id="zed_front_left_img",
                        timestamp=rospy.Time.now()
                    )
                self.aruco_id_pose_publisher.publish_output(
                    relative_pose_msg=relative_pose_encoded
                )

        return


if __name__ == '__main__':
    # Camera name
    CAMERA_NAME = "zed_front"
    CAMERA_INFO_TOPIC = f"/{CAMERA_NAME}/camera_info"

    # Visualizer output topics
    RGB_IN_TOPIC = "/zed_front/image/compressed"
    RGB_OUT_TOPIC = "/aruco_reader/zed_front/id_pose/image/compressed"

    # Service call topic to initialize the aruco pose
    SERVICE_TOPIC = "/initialize_aruco_pose"

    # Localization topics
    LOCALIZATION_TOPIC = "/Localization/odometry/filtered_map"
    TF_STATIC_TOPIC = "/tf_static"


    RATE_HZ = 10  # replace with your desired publishing rate.

    # Initialize the ArUco app ROS node.
    aruco_reader = App(
        camera_name=CAMERA_NAME,
        camera_info_topic=CAMERA_INFO_TOPIC,
        rgb_in_topic=RGB_IN_TOPIC,
        rgb_out_topic=RGB_OUT_TOPIC,
        rate_hz=RATE_HZ,
        service_topic = SERVICE_TOPIC,
        localization_topic = LOCALIZATION_TOPIC,
        tf_static_topic = TF_STATIC_TOPIC
    )

    # Run the ArUco reader node.
    aruco_reader.run()