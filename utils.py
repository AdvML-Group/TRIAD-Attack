import numpy as np
import os
import csv
from PIL import Image
from natsort import index_natsorted

def load_npz_images(npz_path, csv_path, image_root, image_size, batch_size):
    data = np.load(npz_path)
    adv_images = data['arr_0']

    image_ids, true_labels = [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(row['ImageId'])
            true_labels.append(int(row['TrueLabel']) - 1)

    sorted_idx = index_natsorted(image_ids)
    image_ids = [image_ids[i] for i in sorted_idx]
    true_labels = [true_labels[i] for i in sorted_idx]

    ori_images = []
    for image_id in image_ids:
        img = Image.open(os.path.join(image_root, image_id + '.png')).convert('RGB')
        img = img.resize((image_size, image_size), Image.LANCZOS)
        ori_images.append(np.array(img))

    ori_images = np.array(ori_images).astype(np.float32) / 255.0
    adv_images = adv_images.astype(np.float32)
    true_labels = np.array(true_labels)

    for i in range(0, len(ori_images), batch_size):
        yield (
            ori_images[i:i+batch_size],
            adv_images[i:i+batch_size],
            true_labels[i:i+batch_size]
        )
