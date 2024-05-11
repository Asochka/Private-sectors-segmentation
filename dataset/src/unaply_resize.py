import os

images_dirs = ['data/train/images', 'data/black_white', 'data/labels']

for images_dir in images_dirs:

    files = os.listdir(images_dir)

    for file in files:
        if 'resized' in file:
            file_path = os.path.join(images_dir, file)
            os.remove(file_path)
            print(f"Deleted '{file_path}'")