import os
import rasterio
import geopandas as gpd
import torch.nn.functional as F
import numpy as np

from shapely.geometry import shape as Shape
from rasterio.features import shapes
from ultralytics import YOLO
from torch import unsqueeze, squeeze
from torch.nn.functional import interpolate

from src.config import conf_threshold, iou_threshold, agnostic_nms, max_detections


class GEOTIFPredictor:
    def __init__(self, output_format='gpkg', model_name='/Users/21010992/Downloads/miem/runs/segment/train10/weights/best.pt'):
        self.model = YOLO(model=model_name)
        self.model.overrides['conf'] = conf_threshold  # NMS confidence threshold
        self.model.overrides['iou'] = iou_threshold  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = agnostic_nms  # NMS class-agnostic
        self.model.overrides['max_det'] = max_detections  # maximum number of detections per image

        self.output_format = output_format

    def process_image(self, img_path):
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()

            # Apply YOLO model
            results = self.model.predict(img_path)
            if not results[0].masks:
                print(f'No buildings found in image: {img_path}')
                return
            seg = results[0].masks.data
            seg = seg.unsqueeze(1)  # Change size to [118, 1, 640, 640]
            scaled_seg = F.interpolate(seg, size=(meta['height'], meta['width']), mode='nearest')
            scaled_seg = scaled_seg.squeeze(1)
            segs = scaled_seg  # torch.Size([21, 1, 1, 640, 640])

            polygons = []
            for i in range(segs.shape[0]):
                layer = segs[i].numpy().astype(np.uint8)
                for shape, value in shapes(layer, mask=None, transform=src.transform):
                    if value == 1:
                        polygons.append(shape)

            gdf = gpd.GeoDataFrame({'geometry': [Shape(polygon) for polygon in polygons]})
            gdf.crs = src.crs
            return gdf

    def process_image_optimized(self, img_path):
        with rasterio.open(img_path) as src:
            meta = src.meta.copy()

            # Apply YOLO model
            results = self.model.predict(img_path)
            if not results[0].masks:
                print(f'No private areas in image: {img_path}')
                return

            seg = unsqueeze(results[0].masks.data, 1)  # Change size to [118, 1, 640, 640]
            scaled_seg = interpolate(seg, size=(meta['height'], meta['width']), mode='nearest')
            segs = squeeze(scaled_seg, 1)

            polygons = []
            for i in range(segs.shape[0]):
                layer = segs[i].numpy().astype(np.uint8)
                for shape, value in shapes(layer, mask=None, transform=src.transform):
                    if value == 1:
                        polygons.append(shape)

            gdf = gpd.GeoDataFrame({'geometry': [Shape(polygon) for polygon in polygons]})
            gdf.crs = src.crs
            return gdf

    def predict_tiff(self, tif_path, save_dir, extend_name=True):
        gdf = self.process_image(tif_path)
        dirname = ''
        if extend_name:
            dirname = os.path.dirname(tif_path).split('/')[-1]+'_'
        basename = dirname+os.path.basename(tif_path).split('.')[0]
        if gdf is not None:
            output_filename = save_dir + f'{basename}_seg.{self.output_format}'
            if self.output_format == 'gpkg':
                print(os.path.join(save_dir, output_filename))
                gdf.to_file(os.path.join(save_dir, output_filename), driver='GPKG')
            elif self.output_format == 'shp':
                gdf.to_file(os.path.join(save_dir, output_filename))
            else:
                print('ERR: Invalid format.')
        else:
            print('ERR: Empty gdf')
