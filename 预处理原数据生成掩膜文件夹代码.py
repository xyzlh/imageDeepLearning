
import numpy as np
import pandas as pd
import os
import json
import pprint
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import matplotlib.patches as patches
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
with open(
        '../archive/train/_annotations.coco.json',
        'r'
) as train_file:
    train_data = json.load(train_file)
def create_mask(image_path, data, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_name = os.path.basename(image_path)
    image_id = None
    width, height = None, None

    for image_info in data['images']:
        if image_info['file_name'] == file_name:
            image_id = image_info['id']
            width = image_info['width']
            height = image_info['height']
            break

    if image_id is None:
        print(f"Image {file_name} not found in dataset.")
        return

    mask = np.zeros((height, width), dtype=np.uint8)

    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            for segmentation in annotation['segmentation']:
                polygon = np.array(segmentation).reshape(-1, 2)
                cv2.fillPoly(mask, [polygon.astype(np.int32)], color=1)

    mask_path = os.path.join(output_dir, file_name)
    Image.fromarray(mask * 255).save(mask_path)
    # print(f"Mask saved to {mask_path}")
import shutil
def get_all_mask_imgs(data, mask_output_dir, img_output_dir, origin_img_dir):
    images = data['images']
    annotations = data['annotations']

    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)

    for img in images:
        img_path = os.path.join('../archive', img['file_name'])
        create_mask(img_path, data, mask_output_dir)
        origin_img_path = os.path.join(origin_img_dir, img['file_name'])
        new_img_path = os.path.join(img_output_dir, os.path.basename(origin_img_path))
        shutil.copy2(origin_img_path, new_img_path)

with open(
        "../archive/test/_annotations.coco.json",
        "r"
) as test_file:
    test_data = json.load(test_file)

with open(
        "../archive/valid/_annotations.coco.json",
        "r"
) as valid_file:
    valid_data = json.load(valid_file)

get_all_mask_imgs(train_data, '../archive/train_mask', '../archive/train_img',
                  '../archive/train')
get_all_mask_imgs(test_data, '../archive/test_mask', '../archive/test_img',
                  '../archive/test')
get_all_mask_imgs(valid_data, '../archive/valid_mask', '../archive/valid_img',
                  '../archive/valid')