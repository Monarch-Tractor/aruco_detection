import cv2
import numpy as np


class Camera:
    """
    This class represents the camera with its 
    intrinsic and distortion parameters.
    """
    def __init__(self,
                 camera_matrix,
                 dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs


class ArUcoGridGenerator:
    """
    This class generates a grid of ArUco markers 
    on a white background.
    """
    def __init__(self,
                 grid_size,
                 marker_length,
                 camera,
                 padding=50):
        self.grid_size = grid_size
        self.marker_length = marker_length
        self.camera = camera
        self.padding = padding
        self.aruco_dict = \
            cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_6X6_250
            )

    def generate_grid_image(self,
                            image_width,
                            image_height):
        """
        This method creates an image with a grid of 
        ArUco markers aligned to fill the canvas.

        Inputs:
            - image_width: Width of the output image.
            - image_height: Height of the output image.

        Output:
            - image: The generated image with ArUco 
                markers on a white background.
        """
        # Create a white background.
        image = np.ones(
            (image_height, image_width),
            dtype=np.uint8
        ) * 255

        # Calculate marker and spacing dimensions
        # in pixels.
        marker_pixel_size = min(image_width, image_height) // (
            self.grid_size[0] + 1
        )
        spacing_pixel = marker_pixel_size // 2

        # Starting position after adding padding
        x_offset_start = self.padding
        y_offset_start = self.padding

        # Loop over the grid and place markers.
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                # Generate marker ID.
                marker_id = (row * self.grid_size[1]) + col

                # Create a blank canvas for the marker.
                marker_image = np.zeros(
                    (marker_pixel_size, marker_pixel_size),
                    dtype=np.uint8
                )
                cv2.aruco.generateImageMarker(
                    self.aruco_dict,
                    marker_id,
                    marker_pixel_size,
                    marker_image
                )

                # Calculate the marker's position on
                # the grid.
                x_offset = x_offset_start + (
                    col * (marker_pixel_size + spacing_pixel)
                )
                y_offset = y_offset_start + (
                    row * (marker_pixel_size + spacing_pixel)
                )

                # Ensure the marker fits within the image boundaries.
                if ((x_offset + marker_pixel_size > image_width) or \
                    (y_offset + marker_pixel_size > image_height)):
                    continue  # Skip if marker goes out of bounds

                # Place the marker onto the white
                # background.
                image[
                    y_offset:y_offset + marker_pixel_size,
                    x_offset:x_offset + marker_pixel_size
                ] = marker_image

        return image


class ArUcoProcessor:
    """
    This class helps in the creation of 
    an ArUco marker grid.
    """
    def __init__(self,
                 camera,
                 grid_size=(6, 6),
                 marker_length=0.05):
        self.camera = camera
        self.grid_size = grid_size
        self.marker_length = marker_length
        self.grid_generator = ArUcoGridGenerator(
            grid_size=grid_size,
            marker_length=marker_length,
            camera=camera
        )

    def process(self,
                image_width=1280,
                image_height=720):
        """
        This method generates a grid image with 
        the specified dimensions.

        Inputs:
            - image_width: Width of the output image.
            - image_height: Height of the output image.

        Output:
            - The generated grid image.
        """
        return self.grid_generator.generate_grid_image(
            image_width=image_width,
            image_height=image_height
        )


# Main script
if __name__ == "__main__":
    # Camera parameters
    camera_matrix = np.array([
        [1093.27, 0, 965.0],
        [0, 1093.27, 569.0],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32
    )

    # Initialize the camera and processor.
    camera = Camera(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )
    processor = ArUcoProcessor(camera=camera)

    # Generate the grid image.
    grid_image = processor.process(
        image_width=1280,
        image_height=720
    )

    # Display and save the image.
    cv2.imshow("ArUco Grid", grid_image)
    cv2.imwrite("aruco_grid.png", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
