import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import re


class llDataset(Dataset):
    def __init__(
            self,
            image_size=(480, 720)
    ):
        self.ll_image_paths = []
        self.rgb_image_paths = []
        self.depth_image_paths = []

        self.ll_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.depth_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.length = 0
        self.idxs = []

    def _get_len(self):
        self.length = len(self.ll_image_paths)
        self.idxs = list(range(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # If idx is a list or a slice, return a list of items
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        elif isinstance(idx, slice):
            return [self.__getitem__(i) for i in self.idxs[idx]]

        # Low-light image, image , depth
        ll_image_path = self.ll_image_paths[idx]
        image_path = self.rgb_image_paths[idx]
        depth_path = self.depth_image_paths[idx]

        # Load low-light image
        ll_image = Image.open(ll_image_path).convert('RGB')
        print(f"ll shape: {ll_image.size}")
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        ll_image = self.ll_image_transform(ll_image).clamp(-1, 1)
        image = self.image_transform(image).clamp(-1, 1)
        depth = self.depth_transform(depth).clamp(-1, 1)
        mask = None
        return ll_image, image, depth, mask


if __name__ == "__main__":
    dataset = llDataset()
    for i in range(10):
        ll, rgb, depth = dataset[i]
