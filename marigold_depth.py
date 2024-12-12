import diffusers
import torch
from torchvision.utils import save_image

import shutil

import rawpy
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import cv2
import torch
import os
import torch
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import cv2

from dataset.SintelDataset import SintelDataset
from depth_anything_v2.dpt import DepthAnythingV2


def save_single_fig(predict_depth, save_root, id):
    predict_depth = (predict_depth - np.min(predict_depth)) / (np.max(predict_depth) - np.min(predict_depth))
    predict_depth = predict_depth * 255.0
    predict_depth = predict_depth.astype(np.uint8)
    save_path = os.path.join(save_root, f"{id}.png")
    print(f"Saving to {save_path}")
    cv2.imwrite(save_path, predict_depth)


def worker(rank, world_size, encoder, model_configs, dataset, save_root):
    # Setup the device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
    ).to(device)

    # Load the model
    # model = DepthAnythingV2(**model_configs[encoder])
    # model.load_state_dict(
    #     torch.load(f'/dataset/vfayezzhang/test/DIR/main/checkpoints/depth_anything_v2_{encoder}_{rank}.pth',
    #                map_location='cpu', weights_only=True))
    # model = model.to(device).eval()
    with torch.no_grad():
        for idx, (image, depth_gt) in enumerate(dataset):
            if (idx + world_size) % world_size != rank:
                continue
            image, depth_gt = image.to(device), depth_gt.to(device)
            image = image.unsqueeze(0)
            h, w = image.shape[-2:]

            image_numpy = image.squeeze().cpu().numpy().transpose(1, 2, 0)

            prediction = pipe(image, output_type='pt').prediction

            prediction = prediction[..., :h, :w]
            depth = prediction
            # print(f"Depth prediction shape: {depth.shape}")
            if idx % 100 == 0:
                print(f"Having processed {idx} images")
            predict_depth_np = depth.squeeze().cpu().numpy()
            depth_gt_np = depth_gt.squeeze().cpu().numpy()

            save_single_fig(predict_depth_np, save_root, idx)
            # save_fig(image_numpy, predict_depth_np, depth_gt_np, save_root, idx)


def get_dataset(dataset_name):
    # if dataset_name == "Hypersim":
    #     return HypersimDataset()
    if dataset_name == "Sintel":
        return SintelDataset()


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    dataset_name = 'Sintel'
    dataset = get_dataset(dataset_name)
    save_root = '/dataset/vfayezzhang/test/depth-pro/infer/marigold/'
    save_root = os.path.join(save_root, dataset_name)
    os.makedirs(save_root, exist_ok=True)

    print(f"Length of dataset: {len(dataset)}")
    encoder = 'vitl'

    world_size = 2
    print(f'World size: {world_size}')

    mp.spawn(worker, args=(world_size, encoder, model_configs, dataset, save_root), nprocs=world_size,
             join=True)
