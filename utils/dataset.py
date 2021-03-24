import json, os, sys
import numpy as np

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import patches

import torch
from torch.utils.data import Dataset

import rasterio


class SpaceNetDataset(Dataset):

    def __init__(self, geojson_dir, image_dir, transform=None):
        self.geojson_dir = geojson_dir
        self.image_dir = image_dir
        self.transform = transform

        self.geojsons = sorted(os.listdir(geojson_dir), key=lambda x:int(x[x.index('_img')+4:-8]))
        self.images = sorted(os.listdir(image_dir), key=lambda x:int(x[x.index('_img')+4:-4]))

        assert all([geojson.split('_')[4].split('.')[0] == image.split('_')[4].split('.')[0] for geojson, image in zip(self.geojsons, self.images)]), 'geojson and image files are not matched!'

        print(f'SpaceNetDataset loaded : {len(self.images)}')
        
        
    def __getitem__(self, idx):
        geojson_dir = os.path.join(self.geojson_dir, self.geojsons[idx])
        img_dir = os.path.join(self.image_dir, self.images[idx])

        self.file_name = self.geojsons[idx][self.geojsons[idx].index('AOI'):dataset.geojsons[idx].index('.geojson')]

        json_file = self.load_json(geojson_dir)
        img_file = self.load_tif(img_dir)

        self.width = img_file.width
        self.height = img_file.height

        img = img_file.read()
        img = img/np.max(img)
        
        polygons = self.get_buildings(json_file, img_file)
        mask = self._mask(polygons)

        if self.transform:
            img = self.transform(img)
            
        return img, mask, polygons

    def __len__(self):
        return len(self.geojsons)

    def load_json(self, path):
        with open(path) as f:
            json_data = json.load(f)
        return json_data
    
    def load_tif(self, path):
        return rasterio.open(path)

    def _mask(self, polygons):
        img = Image.new('L', (self.height, self.width), 0)
        for i in range(len(polygons)):
            ImageDraw.Draw(img).polygon(polygons[i], outline=1, fill=1)
        mask = np.array(img)
        return mask

    def get_buildings(self, json_data, tif_data):
        factor_x = self.width / (tif_data.bounds.right - tif_data.bounds.left)
        factor_y = self.height / (tif_data.bounds.top - tif_data.bounds.bottom)
        polygons = []
        for ply in json_data['features']:
            poly = []
            if ply['geometry']['type'] == 'Polygon':
                for coord in ply['geometry']['coordinates']:        
                    for x,y,z in coord:
                        poly.append((min((x-tif_data.bounds.left)*factor_x, tif_data.width), min((tif_data.bounds.top-y)*factor_y, tif_data.height)))
                        
            elif ply['geometry']['type'] == 'MultiPolygon':
                for coords in ply['geometry']['coordinates']:
                    for coord in coords:
                        for x,y,z in coord:
                            poly.append((min((x-tif_data.bounds.left)*factor_x, tif_data.width), min((tif_data.bounds.top-y)*factor_y, tif_data.height)))
            else:
                tp = ply['geometry']['type']
                raise Exception(f'The type {tp} is not supported') 
            polygons.append(poly)
        return polygons

    def show_image(self, idx):
        img, _, polygons = self[idx]
        
        fig = plt.figure(figsize=(8,4), dpi=150)
        ax = fig.add_subplot(1,2,1)
        ax.set_xticks(range(0, self.width, 100))
        ax.set_yticks(range(0, self.height, 100))
        ax.set_title(self.file_name, fontsize=7)
        ax.imshow(img.transpose(1,2,0))

        ax = fig.add_subplot(1,2,2)
        ax.set_xticks(range(0, self.width, 100))
        ax.set_yticks(range(0, self.height, 100))
        ax.set_title(self.file_name, fontsize=7)
        ax.imshow(img.transpose(1,2,0))
        
        for i in range(len(polygons)):
            p = np.array(polygons[i])
            ax.plot(p[:,0],p[:,1])
        plt.show()