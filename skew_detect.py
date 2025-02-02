import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
import argparse
import cv2
import sys


class SkewDetect:
    piby4 = np.pi / 4

    def __init__(
        self,
        image_array=None,  # Accept a NumPy array instead of a file path
        sigma=3.0,
        display_output=None,
        num_peaks=20,
        plot_hough=None,
    ):
        self.image_array = image_array  # Store the NumPy array
        self.sigma = sigma
        self.display_output = display_output
        self.num_peaks = num_peaks
        self.plot_hough = plot_hough

    def write_to_file(self, wfile, data):
        for d in data:
            wfile.write(f"{d}: {str(data[d])}\n")
        wfile.write("\n")

    def get_max_freq_elem(self, arr):
        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1
        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]
        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)
        return max_arr

    def display_hough(self, h, a, d):

        plt.imshow(
            np.log(1 + h),
            extent=[np.rad2deg(a[-1]), np.rad2deg(a[0]), d[-1], d[0]],
            cmap=plt.cm.gray,
            aspect=1.0 / 90,
        )
        plt.show()

    def compare_sum(self, value):
        return 44 <= value <= 46

    def display(self, data):
        for i in data:
            print(f"{i}: {str(data[i])}")

    def calculate_deviation(self, angle):
        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetect.piby4 - angle_in_degrees)
        return deviation

    def determine_skew(self):
        """
        Determine the skew angle of the image.
        :return: Dictionary containing skew detection results.
        """
        img = self.image_array  # Use the stored NumPy array
        if img is None:
            raise ValueError("Image array is not provided.")

        # Convert to grayscale if necessary
        if len(img.shape) == 3:  # RGB image
            img = rgb2gray(img)

        edges = canny(img, sigma=self.sigma)
        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)

        if len(ap) == 0:
            return {"Message": "Bad Quality"}

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:
            deviation_sum = int(90 - ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue
            deviation_sum = int(ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue
            deviation_sum = int(-ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue
            deviation_sum = int(90 + ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0
        maxi = 0
        for j in range(len(angles)):
            l = len(angles[j])
            if l > lmax:
                lmax = l
                maxi = j

        if lmax:
            ans_arr = self.get_max_freq_elem(angles[maxi])
            ans_res = np.mean(ans_arr)
        else:
            ans_arr = self.get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        data = {
            "Average Deviation from pi/4": average_deviation,
            "Estimated Angle": ans_res,
            "Angle bins": angles,
        }

        if self.display_output:
            self.display(data)

        if self.plot_hough:
            self.display_hough(h, a, d)

        return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Skew Detection in Images")
    parser.add_argument(
        "-s",
        "--sigma",
        default=3.0,
        dest="sigma",
        help="Sigma for Canny Edge Detection",
        type=float,
    )
    parser.add_argument(
        "-n",
        "--num",
        default=20,
        dest="num_peaks",
        help="Number of Hough Transform peaks",
        type=int,
    )
    parser.add_argument(
        "-p", "--plot", default=None, dest="plot_hough", help="Plot the Hough Transform"
    )
    args = parser.parse_args()

    # Example usage for testing (optional)

    if len(sys.argv) < 2:
        print("Usage: python skew_detect.py <image_path>")
        sys.exit(1)

    # Load the image as a NumPy array
    image_path = sys.argv[1]
    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image_array is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)

    skew_obj = SkewDetect(
        image_array=image_array,
        sigma=args.sigma,
        display_output=True,
        num_peaks=args.num_peaks,
        plot_hough=args.plot_hough,
    )
    result = skew_obj.determine_skew()
    print(result)
