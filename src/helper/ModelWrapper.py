import numpy as np

from abc import ABC, abstractmethod

from tqdm import tqdm

import torch

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

import torchvision.transforms as T

from sentence_transformers import SentenceTransformer, util

from transformers import ViTFeatureExtractor, ViTModel
from transformers import AutoImageProcessor, AutoModel

from PIL import Image

import torch.nn as nn
from transformers import AutoProcessor, CLIPModel



class ModelWrapper(ABC):
    
    @abstractmethod
    def extract_features(img: np.array):
        pass
    
    
class VGG16Extractor(ModelWrapper):
    name = 'vgg16'
    
    def __init__(self):      
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        self.preprocess_input = vgg16_preprocess_input
        
    def __preprocess_image(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        
        res = []
        for img in imgs :
            img = img.resize((224, 224)).convert('RGB')
            x = image.img_to_array(img)
            res.append(x)

        return self.preprocess_input(np.array(res))
    
    @staticmethod
    def extract_features(img: np.array):
        extractor = VGG16Extractor()
        x = extractor.__preprocess_image(img)
        feature = extractor.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
    
class VGG19Extractor(ModelWrapper):
    name = 'vgg19'
    
    def __init__(self):      
        base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        self.preprocess_input = vgg19_preprocess_input
        
    def __preprocess_image(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        
        res = []
        for img in imgs :
            img = img.resize((224, 224)).convert('RGB')
            x = image.img_to_array(img)
            res.append(x)

        return self.preprocess_input(np.array(res))
    
    @staticmethod
    def extract_features(img: np.array):
        extractor = VGG19Extractor()
        x = extractor.__preprocess_image(img)
        feature = extractor.model.predict(x)
        return feature / np.linalg.norm(feature)
    
class CLIPExtractor(ModelWrapper):
    name = 'clip_L14'       
    
    def __init__(self):      
        self.model = SentenceTransformer('clip-ViT-L-14')
    
    @staticmethod
    def extract_features(img):
        extractor = CLIPExtractor()
        return extractor.model.encode(img)
    
class CLIPExtractor_336(ModelWrapper):
    name = 'clip_L14_336'       
    
    def __init__(self):    
        self.device = torch.device('cuda' if torch.cuda.is_available() else "mps")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(self.device)
    
    @staticmethod
    def extract_features(imgs):
        extractor = CLIPExtractor_336()
        
        if not isinstance(imgs, list):
            imgs = [imgs]
        
        with torch.no_grad():
            image_features_list = []
            for img in tqdm(imgs):
                inputs = extractor.processor(images=img, return_tensors="pt").to(extractor.device)
                image_features = extractor.model.get_image_features(**inputs)
                image_features_list.append(image_features)

            image_features_array = torch.cat(image_features_list, dim=0).cpu().numpy()
        
        return image_features_array
    
class DinoV2Extractor(ModelWrapper):
    name = 'dino_v2_giant'       
    
    def __init__(self):    
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
        self.model = AutoModel.from_pretrained("facebook/dinov2-giant").to(self.device)
    
    @staticmethod
    def extract_features(imgs):
        extractor = DinoV2Extractor()
        
        if not isinstance(imgs, list):
            imgs = [imgs]
        
        with torch.no_grad():
            image_features_list = []
            for img in tqdm(imgs):
                inputs = extractor.processor(images=img, return_tensors="pt").to(extractor.device)
                outputs = extractor.model(**inputs)
                image_features = outputs.last_hidden_state.mean(dim=1)
                image_features_list.append(image_features)
                
            image_features_array = torch.cat(image_features_list, dim=0).cpu().numpy()
        
        return image_features_array
        
class ViTExtractor(ModelWrapper):
    name = 'vit_huge'       
    
    def __init__(self):      
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-huge-patch14-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
        
    def __preprocess_image(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        
        res = []
        for img in imgs :
            img = img.resize((224, 224)).convert('RGB')
            x = image.img_to_array(img)
            res.append(x)

        return np.array(res)

    @staticmethod
    def extract_features(img):
        extractor = ViTExtractor()
        
        imgs = extractor.__preprocess_image(img)
        
        features = []
        for img in imgs:
            inputs = extractor.feature_extractor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = extractor.model(**inputs)
            last_hidden_states =  outputs.last_hidden_state            
            features.append(np.squeeze(last_hidden_states.numpy()).flatten())
            
        if len(features) == 1:
            return features[0]
        else:            
            return features
        