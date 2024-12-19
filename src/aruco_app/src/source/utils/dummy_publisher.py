#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import numpy as np


class DummyLocalizationPublisher:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("dummy_localization_publisher", anonymous=True)

        # Publisher for localization data
        self.localization_pub = rospy.Publisher("/localization", PoseStamped, queue_size=10)

        # TF broadcaster for static transforms
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Set publishing rate
        self.rate = rospy.Rate(10)  # 10 Hz

        # Publish a dummy static transform
        self.publish_static_tf()

    def publish_static_tf(self):
        """
        Publish a static transform for `tf_static`.
        """
        static_transform = TransformStamped()

        # Header information
        static_transform.header.stamp = rospy.Time.now()
        static_transform.header.frame_id = "map"  # Parent frame
        static_transform.child_frame_id = "base_link"  # Child frame

        # Transformation values (position and orientation)
        static_transform.transform.translation.x = 1.0
        static_transform.transform.translation.y = 2.0
        static_transform.transform.translation.z = 0.0

        # Quaternion (representing no rotation)
        q = self.euler_to_quaternion(0, 0, 0)  # Roll, Pitch, Yaw
        static_transform.transform.rotation.x = q[0]
        static_transform.transform.rotation.y = q[1]
        static_transform.transform.rotation.z = q[2]
        static_transform.transform.rotation.w = q[3]

        # Broadcast the static transform
        self.tf_broadcaster.sendTransform(static_transform)

        rospy.loginfo("Published static transform from 'map' to 'base_link'.")

    def publish_localization(self):
        """
        Publish dummy localization data.
        """
        pose = PoseStamped()

        # Header information
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        # Position (randomized for demonstration purposes)
        pose.pose.position.x = np.random.uniform(0, 10)
        pose.pose.position.y = np.random.uniform(0, 10)
        pose.pose.position.z = 0.0

        # Orientation (static quaternion for now)
        q = self.euler_to_quaternion(0, 0, np.pi / 4)  # Roll, Pitch, Yaw
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        # Publish the pose
        self.localization_pub.publish(pose)
        rospy.loginfo(f"Published localization: {pose}")

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion.
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return [qx, qy, qz, qw]

    def run(self):
        """
        Main loop to publish localization at a fixed rate.
        """
        while not rospy.is_shutdown():
            self.publish_localization()
            self.rate.sleep()


if __name__ == "__main__":
    try:
        publisher = DummyLocalizationPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down Dummy Localization Publisher.")
