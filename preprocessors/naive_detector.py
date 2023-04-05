import cv2
import numpy as np
import os
import torch
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, disk, binary_closing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from typing import Optional, List, Tuple

class ImagePreprocessor:
    def __init__(self, target_size: tuple = (512, 512)):
        """
            Initialize the ImagePreprocessor class with a target size.

            :param target_size: Tuple with the target height and width of the image. Defaults to (512, 512)
        """
        self.target_size = target_size

    def apply_clahe(self, img: np.ndarray, clip_limit: float = None, tile_grid_size: tuple = (3, 3)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the input image.

        :param img: Input image as a NumPy array.
        :param clip_limit: Clip limit for CLAHE. If not provided, an adaptive clip limit is computed. Defaults to None.
        :param tile_grid_size: Tuple with the grid size for the CLAHE algorithm. Defaults to (3, 3).
        :return: Image after applying CLAHE as a NumPy array.
        """
        if clip_limit is None:
            clip_limit = self.compute_adaptive_clip_limit(img)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img_clahe = clahe.apply(img)
        return img_clahe

    def compute_adaptive_clip_limit(self, img: np.ndarray) -> float:
        """
        Compute an adaptive clip limit for the CLAHE algorithm based on the input image's standard deviation.

        :param img: Input image as a NumPy array.
        :return: Adaptive clip limit as a float.
        """
        std_dev = np.std(img)
        return max(1.0, 2 * std_dev / 255)

    def denoise_bilateral(self, img: np.ndarray, diameter: int = 9, sigma_color: float = None, sigma_space: float = None) -> np.ndarray:
        """
        Denoise the input image using a bilateral filter.

        :param img: Input image as a NumPy array.
        :param diameter: Diameter of the pixel neighborhood used during filtering. Defaults to 9.
        :param sigma_color: Filter sigma in the color space. If not provided, it's estimated based on the noise level. Defaults to None.
        :param sigma_space: Filter sigma in the coordinate space. If not provided, it's estimated based on the noise level. Defaults to None.
        :return: Denoised image as a NumPy array.
        """
        if sigma_color is None or sigma_space is None:
            noise_level = self.estimate_noise_level(img)
            sigma_color = sigma_space = noise_level * 75
        img_denoised = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
        return img_denoised

    def estimate_noise_level(self, img: np.ndarray) -> float:
        """
        Estimate the noise level in the input image using the Median Absolute Deviation (MAD) method.

        :param img: Input image as a NumPy array.
        :return: Estimated noise level as a float.
        """
        median = np.median(img)
        mad = np.median(np.abs(img - median))
        return mad / 0.6745

    def adaptive_threshold(self, img: np.ndarray, max_value: int = 255, block_size: Optional[int] = None, c: int = 5) -> np.ndarray:
        """
        Apply adaptive thresholding to the input image.

        :param img: Input image as a NumPy array.
        :param max_value: Maximum value to use with the THRESH_BINARY thresholding type. Defaults to 255.
        :param block_size: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel. If not provided, it's computed adaptively. Defaults to None.
        :param c: Constant subtracted from the mean or weighted mean. Defaults to 5.
        :return: Image after applying adaptive thresholding as a NumPy array.
        """
        if block_size is None:
            block_size = self.compute_adaptive_block_size(img)
        img_thresholded = cv2.adaptiveThreshold(img, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
        return img_thresholded

    def compute_adaptive_block_size(self, img, percentage=5):
        height, width = img.shape
        return int(min(height, width) * percentage / 100) | 1
    
    def canny_edge_detection(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection to the input image.

        :param img: Input image as a NumPy array.
        :return: Image after applying Canny edge detection as a NumPy array.
        """
        # Compute the median of the image
        median = np.median(img)

        # Set the lower and upper thresholds based on the median value
        lower_threshold = int(max(0, (1.0 - 0.33) * median))
        upper_threshold = int(min(255, (1.0 + 0.33) * median))

        edges = cv2.Canny(img, lower_threshold, upper_threshold)
        return edges

    def filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter contours based on their area and circularity.

        :param contours: List of contours as NumPy arrays.
        :return: Filtered list of contours as NumPy arrays.
        """
        filtered_contours = []

        # Compute the mean area of the contours
        mean_area = np.mean([cv2.contourArea(cnt) for cnt in contours])

        # Set the minimum area based on the mean area
        min_area = mean_area * 0.1

        # Set the minimum circularity
        min_circularity = 0.1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            if area > min_area and circularity > min_circularity:
                filtered_contours.append(cnt)

        return filtered_contours
    
    def erode(self, img: np.ndarray, selem_size: int = 3) -> np.ndarray:
        """
        Erode the input image using binary erosion with a disk-shaped structuring element.

        :param img: Input image as a NumPy array.
        :param selem_size: Size of the disk-shaped structuring element. Defaults to 3.
        :return: Eroded image as a NumPy array.
        """
        selem = disk(selem_size)
        img_eroded = binary_erosion(img, selem)
        return (img_eroded * 255).astype(np.uint8)
    
    def isolate_connected_regions(self, img: np.ndarray, selem_size: int = 3) -> np.ndarray:
        """
        Isolate connected regions in the input image using binary closing with a disk-shaped structuring element.

        :param img: Input image as a NumPy array.
        :param selem_size: Size of the disk-shaped structuring element. Defaults to 3.
        :return: Image with connected regions isolated as a NumPy array.
        """
        selem = disk(selem_size)
        img_closed = binary_closing(img, selem)
        connected_regions = img_closed & ~img
        return connected_regions
    
    def apply_watershed(self, img: np.ndarray, distance_transform: bool = True) -> np.ndarray:
        """
        Apply the watershed algorithm to the input image.

        :param img: Input image as a NumPy array.
        :param distance_transform: Whether to perform the distance transform on the image before applying the watershed algorithm. Defaults to True.
        :return: Image after applying the watershed algorithm as a NumPy array.
        """
        # If distance_transform is True, perform the distance transform on the image
        if distance_transform:
            img = ndimage.distance_transform_edt(img)

        # Find the local maxima in the distance-transformed image
        local_maxima = peak_local_max(img, indices=False, min_distance=3)

        # Label the local maxima
        markers = ndimage.label(local_maxima)[0]

        # Apply the watershed algorithm
        labels = watershed(-img, markers, mask=img)

        return labels

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess the input image and convert it to a PyTorch tensor.

        :param image_path: The path to the input image file.
        :return: Preprocessed image as a PyTorch tensor.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, self.target_size)
        img_clahe = self.apply_clahe(img_resized)
        img_denoised = self.denoise_bilateral(img_clahe)
        img_thresholded = self.adaptive_threshold(img_denoised)
        
        connected_regions = self.isolate_connected_regions(img_thresholded)
        eroded_connected_regions = self.erode(connected_regions)
        
        img_processed = img_thresholded | eroded_connected_regions
        img_tensor = F.to_tensor(img_processed)
        
        return img_tensor

    def draw_contours(self, img: np.ndarray, contours: List[np.ndarray], color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 1) -> np.ndarray:
        """
        Draw contours on the input image.

        :param img: Input image as a NumPy array.
        :param contours: List of contours as NumPy arrays.
        :param color: Tuple representing the color of the contours (B, G, R). Defaults to (255, 0, 0).
        :param thickness: Thickness of the contour lines. Defaults to 1.
        :return: Image with contours drawn as a NumPy array.
        """
        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, color, thickness)
        return img_contours
    
    def process_and_save_contours(self, input_folder: str, output_folder: str, image_name: str) -> None:
        """
        Preprocess the input image, detect contours, and save the image with contours in the output folder.

        :param input_folder: The path to the input folder containing the images to be processed.
        :param output_folder: The path to the output folder where the processed images with contours will be saved.
        :param image_name: The name of the image file to be processed.
        """
        input_image_path = os.path.join(input_folder, image_name)
        img_tensor = self.preprocess_image(input_image_path)

        # Load the original image in color and resize it
        original_img = cv2.imread(input_image_path)
        original_img_resized = cv2.resize(original_img, self.target_size)

        img = F.to_pil_image(img_tensor)
        img = np.array(img)

        # Invert the image
        img_inv = cv2.bitwise_not(img)

        # Morphological operations to separate spots
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(img_inv, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Compute distance transform and normalize it
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
        _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

        # Subtract sure regions from the background
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label the markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply the watershed algorithm
        cv2.watershed(original_img_resized, markers)
        original_img_resized[markers == -1] = [255, 0, 0]

        output_image_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + '_contours.png')
        cv2.imwrite(output_image_path, original_img_resized)


    def preprocess_folder(self, input_folder: str, output_folder: str, num_workers: Optional[int] = None) -> None:
        """
        Preprocess all images in the input folder and save the output images with contours in the output folder.

        :param input_folder: The path to the input folder containing the images to be processed.
        :param output_folder: The path to the output folder where the processed images with contours will be saved.
        :param num_workers: The number of parallel workers to use for preprocessing. If None, no parallelism is used. Defaults to None.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_names = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif'))]

        for image_name in image_names:
            self.process_and_save_contours(input_folder, output_folder, image_name)

if __name__ == '__main__':
    input_folder = 'path/to/input/'
    output_folder = 'path/to/output/'

    preprocessor = ImagePreprocessor(target_size=(2048, 2048))
    preprocessor.preprocess_folder(input_folder, output_folder)