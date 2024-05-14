import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np


dataset_dir = 'dataset_sectors'
originals_dir = os.path.join(dataset_dir, 'basic')
labels_dir = os.path.join(dataset_dir, 'labels')

output_dir = 'formatted_dataset'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

for path in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'labels'), exist_ok=True)

file_names = [os.path.splitext(file)[0] for file in os.listdir(originals_dir) if file.endswith('.png') or file.endswith('.PNG')]

train_files, test_files = train_test_split(file_names, test_size=0.2, random_state=42)  
train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)  

def copy_files(files, source_dir, target_dir_images, target_dir_labels):
    for fname in files:
        original_path = os.path.join(source_dir, fname + '.png')
        label_path = os.path.join(labels_dir, fname + '.txt')
        
        shutil.copy(original_path, target_dir_images)
        shutil.copy(label_path, target_dir_labels)

copy_files(train_files, originals_dir, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
copy_files(val_files, originals_dir, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))
copy_files(test_files, originals_dir, os.path.join(test_dir, 'images'), os.path.join(test_dir, 'labels'))

print("Data split and copied successfully.")
