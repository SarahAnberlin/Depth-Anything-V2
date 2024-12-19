import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.BaseDataset import BaseDataset
from dataset.utils import get_hdf5_array
import cv2


class AM2KDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        meta_json = '/dataset/vfayezzhang/dataset/AM-2K/validation/validation_meta.json'
        self.meta_json = meta_json
        self.image_paths = []
        self.trimap_paths = []
        with open(meta_json, "r", encoding="utf-8") as infile:
            for line in infile:
                entry = json.loads(line)
                self.image_paths.append(entry["img_path"])
                self.trimap_paths.append(entry["trimap_path"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''
        idx: list,int
        Return:
            image: torch.Tensor
            depth: torch.Tensor
        '''
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        to_tensor = transforms.ToTensor()

        image_np = cv2.imread(self.image_paths[idx])
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_np = image_np / 255.0
        image = to_tensor(image_np)

        trimap = cv2.imread(self.trimap_paths[idx], cv2.IMREAD_GRAYSCALE)
        trimap = trimap.astype(np.float32) / 255.0
        trimap = np.clip(trimap, 0.0, 1.0)
        trimap = to_tensor(trimap)
        return image, trimap,


def convert_rgb_path_to_trimap_path(rgb_path):
    return rgb_path.replace("original", "trimap").replace("jpg", "png")


def get_meta(meta_json):
    image_root = "/dataset/vfayezzhang/dataset/AM-2K/validation/original/"
    image_paths = []

    for root, dir, files in os.walk(image_root):
        for file in files:
            image_path = os.path.join(root, file)
            if (not os.path.exists(image_path)):
                print(f"File not found: {image_path}")
                continue
            image_paths.append(image_path)

    image_paths = sorted(image_paths)
    cnt = 0
    with open(meta_json, 'w') as f:
        for id, image_path in image_paths:
            cnt += 1
            json.dump({
                'id': cnt,
                'img_path': image_path,
                'trimap_path': convert_rgb_path_to_trimap_path(image_path)
            }, f)
            f.write('\n')


if __name__ == "__main__":
    meta_json = '/dataset/vfayezzhang/dataset/AM-2K/validation/validation_meta.json'

    if not os.path.exists(meta_json):
        get_meta(meta_json=meta_json)

    dataset = AM2KDataset()
    print(f"Dataset length: {len(dataset)}")

    for id, (image, trimap) in enumerate(dataset):
        print(f"Id: {id}, Image shape: {image.shape}, Trimap shape: {trimap.shape}")
