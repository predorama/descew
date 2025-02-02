import numpy as np
from skimage.transform import rotate
from skew_detect import SkewDetect


class Deskew:
    def __init__(
        self, image_array=None, display_image=False, output_file=None, r_angle=0
    ):
        """
        Initialize the Deskew class.
        :param image_array: Numpy array of the image to deskew.
        :param display_image: Whether to display the deskewed image.
        :param output_file: Path to save the deskewed image (optional).
        :param r_angle: Additional rotation angle to apply (in degrees).
        """
        self.image_array = image_array  # Accept a NumPy array
        self.display_image = display_image
        self.output_file = output_file
        self.r_angle = r_angle
        self.skew_obj = SkewDetect(
            image_array=image_array
        )  # Pass the numpy array to SkewDetect

    def deskew(self):
        """
        Deskew the image by detecting the skew angle and rotating it.
        :return: Deskewed image as a NumPy array.
        """
        if self.image_array is None:
            raise ValueError("Image array is not provided.")

        # Detect the skew angle
        res = self.skew_obj.determine_skew()
        angle = res["Estimated Angle"]

        # Correct the rotation angle based on the detected skew
        if angle >= -45 and angle <= 45:
            rot_angle = angle
        elif angle > 45 and angle <= 90:
            rot_angle = angle - 90
        elif angle < -45 and angle >= -90:
            rot_angle = angle + 90

        # Apply the additional rotation angle (r_angle)
        rot_angle += self.r_angle

        # Rotate the image
        rotated = rotate(self.image_array, rot_angle, resize=True)

        # Display the image if requested
        if self.display_image:
            self.display(rotated)

        # Save the image if an output file is provided
        if self.output_file:
            self.saveImage(rotated * 255)

        return rotated

    def saveImage(self, img):
        """
        Save the deskewed image to the specified output file.
        :param img: Deskewed image as a NumPy array.
        """
        if not self.output_file:
            raise ValueError("Output file path is not provided.")
        from skimage import io

        io.imsave(self.output_file, img.astype(np.uint8))

    def display(self, img):
        """
        Display the deskewed image using matplotlib.
        :param img: Deskewed image as a NumPy array.
        """
        import matplotlib.pyplot as plt  # Import matplotlib for visualization <button class="citation-flag" data-index="6">

        plt.imshow(img, cmap="gray")
        plt.axis("off")  # Hide axes for better visualization
        plt.show()

    def run(self):
        """
        Run the deskewing process.
        :return: Deskewed image as a NumPy array.
        """
        if self.image_array is not None:
            return self.deskew()
        else:
            raise ValueError("No image array provided for deskewing.")


if __name__ == "__main__":
    # Example usage for testing (optional)
    import cv2
    import sys

    if len(sys.argv) < 2:
        print("Usage: python deskew.py <image_path>")
        sys.exit(1)

    # Load the image as a NumPy array
    image_path = sys.argv[1]
    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image_array is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    # Initialize and run the Deskew class
    deskew_obj = Deskew(
        image_array=image_array,
        display_image=True,
        output_file="deskewed_output.jpg",
        r_angle=0,
    )
    deskewed_image = deskew_obj.run()

    print("Deskewing completed successfully!")
