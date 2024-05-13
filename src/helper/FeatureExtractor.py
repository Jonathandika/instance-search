import os
import string

import numpy as np

from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter


from typing import List

import cv2

from src.helper.ModelWrapper import ModelWrapper

class FeatureExtractor:
    def __init__(self, model: ModelWrapper):
        self.model = model
        
    def set_model(self, model: ModelWrapper):
        self.model = model
        
    def load_image(self, path: str) -> Image:
        """
            Load an image from the given path (single file).

            Parameters:
            - path (str): The path to the image file.

            Returns:
            - loaded_image (Image): The loaded image.
        """
        img = Image.open(path)
        return img
        
        
    def load_image_batch(self, path: str) -> list:
            """
                Load and preprocess images from a directory (batch).

                Parameters:
                - path (str): The path to the directory containing the images.

                Returns:
                - loaded_images (list): A numpy array containing the loaded images.
            """
            
            imgs = []
            for filename in os.listdir(path):
                img_path = os.path.join(path, filename)
                x = self.load_image(img_path)
                imgs.append(x)
            return imgs
        
        
        
    def crop_image(self, img: Image, bounding_box: list) -> Image:
        """
            Crop an image if needed.
            
            Parameters:
            - img (Image): The input image to be cropped.
            - bounding_box (list): The bounding box coordinates (x, y, w, h) specifying the region to be cropped.
            
            Returns:
            - cropped_img (Image): The cropped image.
        """
        x, y, w, h = bounding_box
        cropped_img = img.crop([x, y, x + w, y + h])
        return cropped_img
    
    def augment_image(self, img: Image, bounding_box: list) -> Image:
        """
            Augment an image if needed.
            
            Parameters:
            - img (Image): The input image to be augmented.
            - bounding_box (list): The bounding box coordinates (x, y, w, h) specifying the region to be augmented.
            
            Returns:
            - augmented_img (Image): The augmented image.
        """
        augmented_imgs = [img]
    
        # Crop image
        if bounding_box is not None:
            cropped_image = self.crop_image(img, bounding_box)
            augmented_imgs.append(cropped_image)
            
        # Change brightness
        for brightness in [-100, 100, 25]:
            brightness_img = img.point(lambda x: x + brightness)
            augmented_imgs.append(brightness_img)
            
            cropped_image = self.crop_image(img, bounding_box)
            brightness_img = cropped_image.point(lambda x: x + brightness)
            augmented_imgs.append(brightness_img)
            
        # Rotate Images
        for angle in [45, 90, 180]:
            rotated_img = img.rotate(angle)
            augmented_imgs.append(rotated_img)
            
            cropped_image = self.crop_image(img, bounding_box)
            rotated_img = cropped_image.rotate(angle)
            augmented_imgs.append(rotated_img)
            
        # # More Augmentation
        augmented_imgs.append(ImageEnhance.Sharpness(img).enhance(10))
        augmented_imgs.append(ImageEnhance.Contrast(img).enhance(2))
        augmented_imgs.append(img.filter(ImageFilter.BLUR))
        augmented_imgs.append(img.filter(ImageFilter.DETAIL))
        augmented_imgs.append(img.filter(ImageFilter.EDGE_ENHANCE))

        augmented_imgs.append(ImageEnhance.Sharpness(cropped_image).enhance(10))
        augmented_imgs.append(ImageEnhance.Contrast(cropped_image).enhance(2))
        augmented_imgs.append(cropped_image.filter(ImageFilter.BLUR))
        augmented_imgs.append(cropped_image.filter(ImageFilter.DETAIL))
        augmented_imgs.append(cropped_image.filter(ImageFilter.EDGE_ENHANCE))

        # Shear Image
        shear_img = img.transform(img.size, Image.AFFINE, (1, 0.5, 0, 0, 1, 0))
        augmented_imgs.append(shear_img)
        
        return augmented_imgs
        
        
    def extract(self, img: Image = None, img_path: str = None, bounding_box=None, aug=False) -> np.ndarray:
        """
            Extract features from a single image.

            Parameters:
            - img (PIL.Image.Image, optional): The input image object. Either `img` or `img_path` must be provided.
            - img_path (str, optional): The path to the input image file. Either `img` or `img_path` must be provided.
            - bounding_box (tuple, optional): The bounding box coordinates (left, upper, width, height) to crop the image. For the query image if needed.
            - aug (bool, optional): Whether to augment the image or not.

            Returns:
            - features (numpy.ndarray): The extracted features from the image.

            Raises:
            - ValueError: If both `img` and `img_path` are None.
        """
        
        # Validate Inputs (either img or img_path must be provided)
        if img is None and img_path is None:
            raise ValueError('Either img or img_path must be provided')
        elif img is None:
            img = self.load_image(img_path)

        if aug == False:
            features = self.model.extract_features(img)
            return features
        
        else:
            augmented_imgs = self.augment_image(img, bounding_box)
            features = self.model.extract_features(augmented_imgs)
            return features

   
    def extract_batch(self, source_path: str, target_path: str): 
        """
            Extract features from a batch of images and save them.

            Parameters:
            - source_path (str): The path to the directory containing the images.
            - target_path (str): The parent path to save the extracted features.
        """
        

        imgs = self.load_image_batch(source_path)
            
        features = self.model.extract_features(imgs)

        save_path = Path(target_path + "features_" + self.model.name)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        for filename, feature in zip(os.listdir(source_path), features):
            feature_path = Path(save_path) / filename.split('.')[0]
            np.save(feature_path, feature)