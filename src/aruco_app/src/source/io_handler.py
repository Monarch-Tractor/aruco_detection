"""
This script defines the input and output structures
for the ArUco application.
"""


class Input:
    """
    This class defines the input structure for the ArUco
    application.
    """
    def __init__(self):
        # RGB data
        self.rgb_data = None
        # Baselink pose
        self.baselink_pose = None
        # Camera pose
        self.camera_pose = None
        # Static transform from baselink frame
        # to camera frame
        self.baselink_to_camera = None

    def update(self,
               rgb_data=None,
               baselink_pose=None,
               camera_pose=None,
               baselink_to_camera=None):
        """
        This method updates the ArUco app's input.
        """
        if rgb_data is not None:
            self.rgb_data = rgb_data
        if camera_pose is not None:
            self.camera_pose = camera_pose
        if baselink_pose is not None:
            self.baselink_pose = baselink_pose
        if baselink_to_camera is not None:
            self.baselink_to_camera = baselink_to_camera
        return

    def reset(self,
              reset_rgb_data=False,
              reset_baselink_pose=False,
              reset_camera_pose=False,
              reset_baselink_to_camera=False,):
        """
        This method resets the ArUco app's input.
        """
        if reset_rgb_data:
            self.rgb_data = None
        if reset_baselink_pose:
            self.baselink_pose = None
        if reset_camera_pose:
            self.camera_pose = None
        if reset_baselink_to_camera:
            self.baselink_to_camera = None
        return


class Output:
    """
    This class defines the output structure for the ArUco
    application.
    """
    def __init__(self):
        # RGB data
        self.rgb_data = None
        self.relative_pose = None
        self.global_pose = None

    def update(self,
               rgb_data,
               relative_pose=None,
               global_pose=None):
        """
        This method updates the ArUco app's output.
        """
        self.rgb_data = rgb_data
        if relative_pose is not None:
            self.relative_pose = relative_pose
        if global_pose is not None:
            self.global_pose = global_pose
        return

    def reset(self):
        """
        This method resets the ArUco app's output.
        """
        self.rgb_data = None
        self.relative_pose = None
        self.global_pose = None
        return
    