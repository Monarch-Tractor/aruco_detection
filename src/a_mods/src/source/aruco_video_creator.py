"""
This script generates a video by applying different camera 
rotations and distances to an input image.
"""


import cv2
import numpy as np

from tqdm import tqdm

from camera import Camera


class ImageTransformer:
    """
    This class is responsible for generating new views of 
    the input image by applying transformations like rotation 
    and translation based on the given distance and angle.
    """
    def __init__(self,
                 camera,
                 in_img):
        """
        Initialize the ImageTransformer with the camera 
        instance and input image.

        Input:
            - camera (Camera): Camera instance that holds 
                  intrinsic and distortion coefficients.
            - in_img (np.ndarray): The input image to transform.
        """
        self.camera = camera
        self.in_img = in_img
        self.height, self.width = in_img.shape[:2]

    def generate_view(self,
                      distance,
                      angle):
        """
        This method generates a new perspective of 
        the input image based on the given distance 
        and angle.

        Input:
            - distance (float): Distance from the camera to 
                  the scene
            - angle (float): Rotation angle of the camera

        Output:
            - transformed_img (np.ndarray): Transformed image
        """
        # Camera rotation (for simplicity, we simulate
        # small rotations around the Z-axis).
        rotation_matrix = cv2.getRotationMatrix2D(
            (self.width // 2, self.height // 2),
            angle,
            1
        )

        # Apply the distortion coefficients to simulate
        # the camera's lens distortion
        distorted_img = cv2.undistort(
            self.in_img,
            self.camera.intrinsic_matrix,
            self.camera.distortion_vector
        )

        # To simulate the effect of distance, we would scale the
        # image (zoom in or out). This simulates the effect of being
        # closer or further away from the scene by resizing the image.
        # Inverse relation between distance and scaling factor
        scaling_factor = 1.0 / distance

        # Resize the image (zoom in if closer, zoom out if farther).
        resized_img = cv2.resize(
            distorted_img,
            None,
            fx=scaling_factor,
            fy=scaling_factor
        )

        # Get the new dimensions of the resized image.
        new_height, new_width = resized_img.shape[:2]

        # Calculate the offset to center the image
        offset_x = (self.width - new_width) // 2
        offset_y = (self.height - new_height) // 2

        # Create a new canvas of the target size (frame size).
        canvas = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        # Place the resized image at the center of the canvas.
        canvas[
            offset_y:offset_y + new_height,
            offset_x:offset_x + new_width
        ] = resized_img

        # Apply the combined affine transformation (rotation)
        transformed_img = cv2.warpAffine(
            canvas,
            rotation_matrix,
            (self.width, self.height)
        )

        return transformed_img


class VideoGenerator:
    """
    This class handles the generation of a video from multiple 
    transformed views of an input image. It applies different 
    camera rotations and translations to create new perspectives
    and writes them to a video file.
    """
    def __init__(self,
                 camera,
                 in_img,
                 output_video_path,
                 angles,
                 distances,
                 fps=10):
        """
        Initialize the VideoGenerator with necessary parameters.

        Input:
            - camera (Camera): Camera instance to use for transformations
            - in_img (np.ndarray): Input image to generate views from
            - output_video_path (str): Path to save the generated video
            - angles (np.ndarray): Array of angles for camera rotation
            - distances (np.ndarray): Array of distances for camera 
                  positioning
            - fps (int, optional): Frame rate of the video (default is 10)
        """
        self.camera = camera
        self.in_img = in_img
        self.output_video_path = output_video_path
        self.angles = angles
        self.distances = distances
        self.fps = fps
        self.generated_views = []
        self.frame_size = (in_img.shape[1], in_img.shape[0])

    def generate_video(self):
        """
        This method generates the video by applying transformations 
        to the input image and writing the frames to the video file.
        """
        transformer = ImageTransformer(
            camera=self.camera,
            in_img=self.in_img
        )

        # Generate multiple views of the input image from
        # different perspectives.
        for distance in tqdm(self.distances, desc="Distances"):
            for angle in tqdm(self.angles, desc="Angles", leave=False):
                view = transformer.generate_view(distance, angle)
                self.generated_views.append(view)

        # # Ensure we have at least 1000 images.
        # while len(self.generated_views) < 1000:
        #     # Duplicate to ensure enough images.
        #     self.generated_views.extend(self.generated_views)

        # # Limit the list to exactly 1000 images.
        # self.generated_views = self.generated_views[:1000]

        # Create a video from the generated images
        self.create_video()

    def create_video(self):
        """
        This method creates a video file from the generated images by
        writing the transformed views to a video file using the OpenCV 
        VideoWriter.
        """
        # Use mp4v codec for .mp4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.fps,
            self.frame_size
        )

        # Write the generated images to the video.
        for view in tqdm(self.generated_views, desc="Writing frames"):
            video_writer.write(view)

        # Release the VideoWriter and close the video.
        video_writer.release()
        print(f"Video saved to {self.output_video_path}")

        return


if __name__ == "__main__":
    # Load the input image
    IN_IMG_PATH = "aruco_grid-in.png"
    in_img = cv2.imread(IN_IMG_PATH)

    # Define angles and distances.
    angles = np.linspace(0, 90, 50)
    distances = np.linspace(1.0, 2.5, 50)

    # Initialize Camera instance.
    # Camera name
    CAMERA_NAME = "zed_front"
    # Camera information ROS topic
    CAMERA_INFO_TOPIC = f"/{CAMERA_NAME}/camera_info"
    # Initialize camera and processor.
    camera = Camera(
        camera_name=CAMERA_NAME,
        camera_info_topic=CAMERA_INFO_TOPIC,
        use_default_intrinsics=True
    )

    # Define the output video path.
    OUT_VID_PATH= "aruco_grid-vid1.mp4"

    # Initialize the VideoGenerator instance
    video_generator = VideoGenerator(
        camera=camera,
        in_img=in_img,
        output_video_path=OUT_VID_PATH,
        angles=angles,
        distances=distances
    )

    # Generate the video
    video_generator.generate_video()
