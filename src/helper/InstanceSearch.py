import os

import cv2
import numpy as np

from src.helper.FeatureExtractor import FeatureExtractor as fe

from tqdm import tqdm

from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch.nn as nn
import torch

class InstanceSearch:
    def __init__(self, config: dict, feature_extractor: fe = None) -> None:
        self.config = config
        
        self.feature_extractor = feature_extractor
    
        self.query_ids = []
        self.query_boxes = {}
        
        # Validate Config
        assert 'query_path' in self.config, 'query_path is required in config'
        assert 'query_box_path' in self.config, 'query_box_path is required in config'
        assert 'gallery_path' in self.config, 'gallery_path is required in config'
        assert 'feature_path' in self.config, 'gallery_path is required in config'
        
        # Get Filenames
        logging.info(f"Loading files from {config['query_path']}")
        self.query_ids = [filename.split(".")[0] for filename in os.listdir(config["query_path"]) if not filename.startswith(".")]
        
        # Load Query Boxes
        logging.info(f"Loading query boxes from {config['query_box_path']}")
        for filename in os.listdir(config["query_box_path"]):
            self.query_boxes[filename.split(".")[0]] = np.loadtxt(os.path.join(config["query_box_path"], filename)).astype(int)
            
    
    def set_feature_extractor(self, feature_extractor: fe):
        self.feature_extractor = feature_extractor
        
    def __cosine_clip(self, x: np.ndarray, y: np.ndarray) -> float:
        x = torch.tensor(x)
        y = torch.tensor(y)
        cos = nn.CosineSimilarity(dim=0)
        sim = cos(x, y).item()
       
        return sim
        

    def get_distance_calculator(self, d: str):
        if d == 'euclidean':
            return distance.euclidean
        elif d == 'cosine':
            return distance.cosine
        elif d == 'cosine_clip':
            return self.__cosine_clip
        else:
            raise ValueError('Invalid distance metric')
    
    
    def search(self, 
               query_id: str, 
               k: int = 10, 
               plot = False, 
               distance='euclidean', 
               query_augmentation: bool = False,
               query_expansion: bool = False,
               query_expansion_2: bool = False) -> list:

        queries = self.feature_extractor.extract(img_path=self.config['query_path'] + query_id + '.jpg', 
                                                 bounding_box=self.query_boxes[query_id],
                                                 aug=query_augmentation)
        
        distances = {}
        feature_path = os.path.join(self.config['feature_path'], ('features_' + self.feature_extractor.model.name))
        
        distance_calculator = self.get_distance_calculator(distance)
        
        for i in tqdm(os.listdir(feature_path)):
            if i.split('.')[-1] != 'npy':
                continue
            feature = np.load(os.path.join(feature_path, i))
            
            d = []
            for query in queries:
                
                d.append(distance_calculator(query, feature))
                                
            distances[i.split('.')[0]] = np.mean(d)

        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        
        if query_expansion:
            
            # Get the second closest image
            second_closest_ids = list(distances.keys())[1:4]
            
            # Query using the second closest image
            second_query = []
            for id in second_closest_ids:
                second_query.extend(self.feature_extractor.extract(img_path=self.config['gallery_path'] + id + '.jpg'))
            
            for i in tqdm(os.listdir(feature_path)):
                
                if i.split('.')[-1] != 'npy':
                    continue
                feature = np.load(os.path.join(feature_path, i))
                
                d = []
                for query in second_query:
                    d.append(distance_calculator(query, feature))
                
                # Get mean distance
                d.append(distances[i.split('.')[0]])
                distances[i.split('.')[0]] = np.mean(d)
                
            distances = dict(sorted(distances.items(), key=lambda item: item[1]))


        if plot:
            
            num_rows = (k + 2) // 12 + 1
            num_cols = 12

            figsize = (num_cols * 2.25, num_rows * 2)
            
            plt.figure(figsize=figsize)
            plt.subplot(num_rows, num_cols, 1)
            plt.imshow(Image.open(self.config['query_path'] + query_id + '.jpg'))
            plt.title(f'Query Image {query_id}')
            plt.axis('off')

            for i, (query_id, dist) in enumerate(distances.items()):
                if i >= k:
                    break
                img = Image.open(self.config['gallery_path'] + query_id + '.jpg')
                plt.subplot(num_rows, 12, i+3)
                plt.imshow(img)
                plt.title(f'{query_id} ({dist:.3f})')
                plt.axis('off')

            plt.show()
            
        return list(distances.keys())[:k]
        
    def search_all(self, 
                   k: int = 10, 
                   plot = True, 
                   query_augmentation: bool = False,
                   query_expansion: bool = False):
        
        result = {}
        for query_id in self.query_ids:
            result[query_id] = self.search(query_id, k, plot, query_augmentation=query_augmentation, query_expansion=query_expansion)

        return result

