"""
This Python script can be used to visualize any vg_msgs/Visualizer  
messages being published to a ROS topic. The script reads the 
messages from the topic, extracts the different 2D and 3D visualizations
and publishes them to the specified topics at a specified rate.
"""

import numpy as np
import cv2

import rospy
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry

from camera import Camera
from aruco_detector import ArUcoProcessor

import tf2_ros
import tf2_geometry_msgs


class ArUcoIDPosePublisher:
    """
    This class listens to the ROS topic publishing
    the images with ArUco markers and publishes the
    IDs of the detected markers along with their poses.
    """
    def __init__(self,
                 camera_name,
                 rgb_out_topic):
        self.camera_name = camera_name
        self.rgb_out_topic = rgb_out_topic
        self.get_publishers()

    def get_publishers(self):
        """
        This method defines the different publishers.
        """
        self.rgb_out_publisher = rospy.Publisher(
            name=self.rgb_out_topic,
            data_class=CompressedImage,
            queue_size=10
        )
        return

    def encode_image(self, img, timestamp):
        """
        This method encodes a numpy array of an image into a 
        sensor_msgs/CompressedImage message.
        """
        # Create a CompressedImage message and publish it.
        msg = CompressedImage()
        msg.header = Header(
            frame_id="map",
            stamp=timestamp,
        )
        _, compressed_img = cv2.imencode(".jpg", img)
        msg.format = "jpeg"
        msg.data = np.asarray(compressed_img).tobytes()
        return msg

    def publish_output(self, rgb_out_msg):
        """
        This method publishes the output of the
        ArUco marker detection and pose estimation
        algorithm.
        """
        # Publish ID and pose of the detected ArUco
        # markers as a compressed image.
        self.rgb_out_publisher.publish(rgb_out_msg)
        return


class ArUcoReader:
    """
    This class reads input ArUco marker images from a 
    ROS bag file, feeds the images to  ArUco marker detection 
    and pose estimation algorithm, and finally publishes the 
    output of the algorithm as a compressed image at a specified
    rate.
    """
    def __init__(self,
                 camera_name,
                 camera_info_topic,
                 rgb_in_topic,
                 rgb_out_topic,
                 rate_hz,
                 service_topic):
        self.camera_name = camera_name
        self.camera_info_topic = camera_info_topic

        self.rgb_in_topic = rgb_in_topic
        self.rgb_out_topic = rgb_out_topic

        self.rate_hz = rate_hz

        self.sevice_topic = service_topic

        # Initialize the ROS node.
        rospy.init_node(
            name="aruco_marker_reader_node",
            anonymous=True
        )

        # Define subscribers.
        self.get_subscribers()

        # # Define the service call.
        # self.set_services()

        # Initialize the tf buffer.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.lookup_static_transform()

        self.aruco_poses = {}

        # Get camera instance.
        self.camera = Camera(
            camera_name=self.camera_name,
            camera_info_topic=self.camera_info_topic
        )

        # Get ArUco processor instance.
        self.aruco_processor = ArUcoProcessor(
            camera=self.camera
        )

        # Publishers for the topics.
        self.aruco_id_pose_publisher = ArUcoIDPosePublisher(
            camera_name=self.camera_name,
            rgb_out_topic=self.rgb_out_topic
        )

        # Input data attribute.
        self.rgb_in = None

        # Output data attribute.
        self.rgb_out = None

    def get_subscribers(self):
        """
        This method defines the different subscribers.
        """
        self.rgb_in_subscriber = rospy.Subscriber(
            name=self.rgb_in_topic,
            data_class=CompressedImage,
            callback=self.rgb_topic_callback
        )

        self.initialize_aruco_pose_service = rospy.Subscriber(
            name=self.service_topic,
            data_class=Bool,
            callback=self.initialize_aruco_pose_callback
        )
        
        self.localization_subscriber = rospy.Subscriber(
            name=self.localization_topic,
            data_class=Odometry,
            callback=self.localization_callback
        )
        return

    def initialize_aruco_pose_service(self, msg):
        """
        This method initializes the ArUco pose.
        """
        if msg.data:

            rospy.loginfo("Initializing ArUco pose.")
            self.aruco_processor.initialize_aruco_pose()
        return

    def decode_image(self, msg):
        """
        This method decodes a sensor_msgs/CompressedImage message
        into a numpy array.
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
        try:
            self.baselink_to_camera = self.tf_buffer.lookup_transform(
                self.baselink_frame, self.camera_frame, rospy.Time(0), rospy.Duration(5.0)
            )
            rospy.loginfo("Static transform from /baselink to /camera obtained.")
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Failed to lookup static transform: {e}")
        return
    
    def localization_callback(self, msg):
        self.baselink_pose = msg.pose.pose
        self.camera_pose = tf2_geometry_msgs.do_transform_pose(self.baselink_pose, self.baselink_to_camera)
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
        self.rgb_in = self.decode_image(msg=rgb_msg)
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
        "Reset the input data attributes."
        self.rgb_in = None
        return

    def process(self):
        """
        This method reads the input data, processes it
        using the ArUco marker detection and pose estimation
        algorithm, and publishes the output.
        """
        if self.rgb_in is not None:
            # Process the input data.
            self.rgb_out = self.aruco_processor.process_image(
                img=self.rgb_in
            )
            rgb_out_encoded = \
                self.aruco_id_pose_publisher.encode_image(
                    img=self.rgb_out,
                    timestamp=rospy.Time.now()
                )
            self.aruco_id_pose_publisher.publish_output(
                rgb_out_msg=rgb_out_encoded
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

    # Initialize the ArUco reader ROS node.
    aruco_reader = ArUcoReader(
        camera_name=CAMERA_NAME,
        camera_info_topic=CAMERA_INFO_TOPIC,
        rgb_in_topic=RGB_IN_TOPIC,
        rgb_out_topic=RGB_OUT_TOPIC,
        rate_hz=RATE_HZ,
        service_topic = SERVICE_TOPIC
    )

    # Run the ArUco reader node.
    aruco_reader.run()