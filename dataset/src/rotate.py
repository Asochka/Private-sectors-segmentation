import cv2
import os

images_dir = 'dataset_sectors/basic'
# images_dir = 'data/black_white'

image_files = [f for f in os.listdir(images_dir)]


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

for img_filename in image_files:
    img_path = os.path.join(images_dir, img_filename)
    img = cv2.imread(img_path)
    
    for i in range(1, 4):
        angle = 90 * i
        rotated_img = rotate_image(img, angle)
    
        new_filename = f"{os.path.splitext(img_filename)[0]}_rotated_{angle}.png"
        new_image_path = os.path.join(images_dir, new_filename)
        
        cv2.imwrite(new_image_path, rotated_img)
        print(f"Saved rotated image as '{new_image_path}'")
