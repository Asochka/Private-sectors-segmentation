import os
import cv2
from PIL import Image
import zipfile


with zipfile.ZipFile("./data/basic.zip","r") as zip_ref:
    zip_ref.extractall("./data/basic")

with zipfile.ZipFile("./data/labels.zip","r") as zip_ref:
    zip_ref.extractall("./data/labels")
    
images_dir_labels = './data/labels'
images_dir_basic = './data/basic'

image_files = [f for f in os.listdir(images_dir_labels)]
for img_filename in image_files:
    img_path = os.path.join(images_dir_labels, img_filename)
    _mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.bitwise_not(_mask)
    cv2.imwrite(img_path, mask)
    image = Image.open(img_path)
    image.save(img_path)
    

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

image_files = [f for f in os.listdir(images_dir_labels)]
for img_filename in image_files:
    img_path = os.path.join(images_dir_labels, img_filename)
    make_non_bw_white(img_path, img_path)
    

image_files = [f for f in os.listdir(images_dir_basic)]
for img_filename in image_files:
    img_path = os.path.join(images_dir_basic, img_filename)
    image = Image.open(img_path).convert("RGB")
    image.save(img_path)
