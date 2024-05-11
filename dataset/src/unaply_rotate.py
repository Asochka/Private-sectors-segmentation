import os

images_dirs = ['dataset_sectors/black&white', 'dataset_sectors/basic']

for images_dir in images_dirs:

    files = os.listdir(images_dir)

    for file in files:
        if 'rotated' in file:
            file_path = os.path.join(images_dir, file)
            os.remove(file_path)
            print(f"Deleted '{file_path}'")
