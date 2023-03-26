import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import Union, List, Tuple
from PIL import Image
# from sklearn.preprocessing import normalize

class SpotDetector:
    def __init__(self) -> None:
        self.current_path = os.getcwd()
        self.files_path = f'{self.current_path}/files'

    def _read_images(self, input_folder: str) -> List[np.ndarray]:
        """
        Read the images from the specified path.

        Args:
            path (str): The path to the folder containing the images.

        Returns:
            List[np.ndarray]: A list of images as NumPy arrays.
        """
        images = []
        for filename in os.listdir(input_folder):
            if filename.endswith('.tiff') or filename.endswith('.tif'):
                img = Image.open(os.path.join(input_folder, filename))
                img = img.convert('L')
                img_np = np.array(img).astype(np.uint8)
                if img_np is not None:
                    img_3_channels = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                    images.append(img_3_channels)
        return images

    def _clahe(self, image: np.ndarray, clip_limit: Union[int, float], grid: tuple) -> np.ndarray:
        """
        Takes grayscale image, clips histogram at a clip limit, uses
        grid for histogram equalization
        Returns CLAHE image as numpy array
        """
        # Check image type
        if not isinstance(image, np.ndarray):
            raise TypeError(f"'image' parameter must be a numpy array, got {type(image)}")
        
        CLAHE = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
        return CLAHE.apply(image)

    def _threshold(self,
                  image: np.ndarray,
                  thresh: int,
                  maxval: int, 
                  thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                  ) -> np.ndarray:
        """
        Defines a threshold for image based on CV2 parameters
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"'image' parameter must be a numpy array, got {type(image)}")
        _, with_threshold = cv2.threshold(image, thresh, maxval, thresh_type)
        return with_threshold

    def _adaptive_threshold(self,
                           image: np.ndarray, 
                           max_value: int, 
                           pixel_neighbors: int,
                           C: int,
                           method = cv2.ADAPTIVE_THRESH_MEAN_C, 
                           thresh_type = cv2.THRESH_BINARY
                           ) -> np.ndarray:
        """
        Implements adaptive threshold instead of forced one
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"'image' parameter must be a numpy array, got {type(image)}")
        return cv2.adaptiveThreshold(image, max_value, method, thresh_type, pixel_neighbors, C)

    def _apply_morphology(self, image: np.ndarray, minimum = (1, 1), maximum = (20, 20)) -> np.ndarray:
        """
        Implements image morphology annotation
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"'image' parameter must be a numpy array, got {type(image)}")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, minimum)
        blob = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, maximum)
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
        blob = 255 - blob
        return blob

    def _get_contours(self, blob: np.ndarray) -> tuple:
        """
        Searches for spots based on morphology annotation        
        """
        if not isinstance(blob, np.ndarray):
            raise TypeError(f"'blob' parameter must be a numpy array, got {type(blob)}")
        cnts, _ = cv2.findContours(blob, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cnts

    def _draw_contours(self, image: np.ndarray, cnts: tuple) -> None:
        """
        Applies the detected contours to the image
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"'image' parameter must be a numpy array, got {type(image)}")
        if not isinstance(cnts, tuple):
            raise TypeError(f"'cnts' parameter must be a numpy array, got {type(cnts)}")
        plt.figure(figsize = (12,12))
        plt.imshow(cv2.drawContours(image, cnts, -1, (0, 255, 0), 1), cmap='gray')
        plt.show()


class ImageProcessor:
    def __init__(self, sd: SpotDetector) -> None:
        self.sd = sd

    def process_image(self, image: np.ndarray) -> None:
        """
        Process an image using the SpotDetector instance.

        Args:
            image (np.ndarray): The input image as a NumPy array.
        """
        with_clahe = self.sd._clahe(image, 2.5, (1,1))
        with_threshold = self.sd._threshold(with_clahe, 0, 255)
        adaptive = self.sd._adaptive_threshold(with_clahe, 255, 75, 5)
        blob = self.sd._apply_morphology(adaptive)
        contours = self.sd._get_contours(blob)
        self.sd._draw_contours(image, contours)


class ImageAligner:
    def __init__(self, sd: SpotDetector, ip: ImageProcessor) -> None:
        self.sd = sd
        self.ip = ip

    def align_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Align a list of images using the DIHE method.

        Args:
            images (List[np.ndarray]): A list of images as NumPy arrays.

        Returns:
            List[np.ndarray]: A list of aligned images as NumPy arrays.
        """
        aligned_images = []
        reference_image = images[0]

        # Pass the first image through the DIHE network
        dihe = cv2.createAlignMTB()
        dihe.process([images[0]], [])
        aligned_images.append(images[0])

        for img in images[1:]:
            # Resize the image to match the reference image
            img = cv2.resize(img, (reference_image.shape[1], reference_image.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Convert the image to the format accepted by DIHE
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = np.uint8(img)

            # Pass the current image through the DIHE network
            aligned_image = np.zeros_like(img)
            dihe.process([img], aligned_image)

            aligned_images.append(aligned_image)

        return aligned_images

    def process_images(self, input_folder: str) -> None:
        images = self.sd._read_images(input_folder)
        if len(images) > 0:
            aligned_images = self.align_images(images)
            for idx, image in enumerate(aligned_images):
                print(f"Processing image {idx + 1}/{len(aligned_images)}")
                self.ip.process_image(image)
        else:
            print("No images found.")
            
    def save_images(self, images: List[np.ndarray], output_folder: str) -> None:
        """
        Save the images to the specified output folder.

        Args:
            images (List[np.ndarray]): A list of images as NumPy arrays.
            output_folder (str): The path to the folder where the images should be saved.
        """
        os.makedirs(output_folder, exist_ok=True)
        for i, img in enumerate(images):
            cv2.imwrite(os.path.join(output_folder, f'aligned_image_{i}.tif'), img)

    def process_images(self, input_folder: str, output_folder: str) -> None:
        """
        Process the images in the input folder and save the aligned images to the output folder.

        Args:
            input_folder (str): The path to the folder containing the input images.
            output_folder (str): The path to the folder where the aligned images should be saved.
        """
        images = self.sd._read_images(input_folder)
        if len(images) > 0:
            aligned_images = self.align_images(images)
            self.save_images(aligned_images, output_folder)
            for idx, image in enumerate(aligned_images):
                print(f"Processing image {idx + 1}/{len(aligned_images)}")
                self.ip.process_image(image)
        else:
            print("No images found.")

    def display_image_pairs(self, raw_folder: str, aligned_folder: str) -> None:
        """
        Display pairs of raw and aligned images side by side.

        Args:
            raw_folder (str): The path to the folder containing the raw images.
            aligned_folder (str): The path to the folder containing the aligned images.
        """
        raw_images = self.sd._read_images(raw_folder)
        aligned_images = self.sd._read_images(aligned_folder)

        if len(raw_images) != len(aligned_images):
            print("The number of raw images and aligned images does not match.")
            return

        num_images = len(raw_images)
        fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 5))

        for i in range(num_images):
            axs[i, 0].imshow(cv2.cvtColor(raw_images[i], cv2.COLOR_BGR2RGB))
            axs[i, 0].set_title(f"Raw Image {i+1}", fontsize=16)
            axs[i, 0].axis('off')

            axs[i, 1].imshow(cv2.cvtColor(aligned_images[i], cv2.COLOR_BGR2RGB))
            axs[i, 1].set_title(f"Aligned Image {i+1}", fontsize=16)
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

        
if __name__ == "__main__":
    sd = SpotDetector()
    ip = ImageProcessor(sd)
    ia = ImageAligner(sd, ip)

    raw_folder = f'{sd.files_path}/raw'
    aligned_folder = f'{sd.files_path}/aligned'

    ia.process_images(raw_folder, aligned_folder)
    ia.display_image_pairs(raw_folder, aligned_folder)
