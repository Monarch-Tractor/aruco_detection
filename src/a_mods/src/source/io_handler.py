class Input:
    """
    This class defines the input structure for the ArUco
    application.
    """
    def __init__(self):
        # RGB data
        self.rgb_data = None

    def update(self,
               rgb_data):
        """
        This method updates the ArUco app's input.
        """
        self.rgb_data = rgb_data
        return

    def reset(self):
        """
        This method resets the ArUco app's input.
        """
        self.rgb_data = None
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
    