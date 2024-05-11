import torch
import torchvision
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from typing import Tuple, List, Dict, Generator, Any
from IPython.display import clear_output


def make_non_bw_white(image_path, output_path):
    with Image.open(image_path) as img:
        img = img.convert('RGBA')
        pixels = img.load()

        width, height = img.size

        for x in range(width):
            for y in range(height):
                r, g, b, a = pixels[x, y]
                if (r, g, b) != (0, 0, 0) and (r, g, b) != (255, 255, 255):
                    pixels[x, y] = (0, 0, 0, a)

        img.save(output_path)


def get_subimages_generator(
    image: Image.Image,
    subimage_size: Tuple[int, int, int]
) -> Generator[Image.Image, None, None]:
  for r in range(image.size[1] // subimage_size[1]):
    for c in range(image.size[0] // subimage_size[0]):
      yield image.crop(box=(
              c * subimage_size[0],
              r * subimage_size[1],
              (c + 1) * subimage_size[0],
              (r + 1) * subimage_size[1]
          )
      )


def save_dataset_subimages(classes_filter: Dict[Tuple[int, int, int], float]):

  for i, filename in enumerate(listdir('basic/')):
      basename = filename[:filename.find('.tif')]

      image = Image.open(fp=f'basic/{basename}.tif').crop(box=(56, 0, 1680 - 56, 1120))
      image_labeled = Image.open(fp=f'labels/{basename}.tif').crop(box=(56, 0, 1680 - 56, 1120))
      subimages = get_subimages_generator(image=image, subimage_size=(224,224))
      subimages_labeleds = get_subimages_generator(image=image_labeled, subimage_size=(224,224))

      for si, subimage in enumerate(subimages):
        subimage_labeled = next(subimages_labeleds)
        subimage.save(fp=f'dataset/originals/i{i}si{si}.tif')
        subimage_labeled.save(fp=f'dataset/labeleds/i{i}si{si}.tif')


def main():
    images_dir = 'labels/'

    image_files = [f for f in os.listdir(images_dir)]
    for img_filename in image_files:
        img_path = os.path.join(images_dir, img_filename)
        _mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.bitwise_not(_mask)
        cv2.imwrite(img_path, mask)
        image = Image.open(img_path)
        image.save(img_path)

    image_files = [f for f in os.listdir(images_dir)]
    for img_filename in image_files:
        img_path = os.path.join(images_dir, img_filename)
        make_non_bw_white(img_path, img_path)

    save_dataset_subimages(
        classes_filter={
            (255, 255, 255): 5
        }
    )

if __name__ == '__main__':
   main()
