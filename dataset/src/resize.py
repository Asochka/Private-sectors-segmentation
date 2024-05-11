import cv2
import os

images_dirs = ['data/train/images', 'data/valid/images', 'data/black_white']
# images_dirs = ['data/valid/images']

for images_dir in images_dirs:

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.PNG')]

    for img_filename in image_files:
        img_path = os.path.join(images_dir, img_filename)
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (1500, 1500), interpolation=cv2.INTER_AREA)
        new_filename = f"{os.path.splitext(img_filename)[0]}_resized.png"
        new_image_path = os.path.join(images_dir, new_filename)
        cv2.imwrite(new_image_path, resized_img)
        