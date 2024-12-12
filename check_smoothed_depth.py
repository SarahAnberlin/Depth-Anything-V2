import shutil

import rawpy
import cv2
import numpy as np
import os

import cv2
import torch
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import cv2
from depth_anything_v2.dpt import DepthAnythingV2


def worker(rank, world_size, encoder, model_configs, files):
    # Setup the device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}_{rank}.pth', map_location='cpu'))
    model = model.to(device).eval()

    files_to_process = files[rank::world_size]

    # Process data
    for file in files_to_process:
        rgb = cv2.imread(file)
        depth = model.infer_image(rgb)
        base_name = os.path.basename(file)
        dir_name = os.path.dirname(file)
        depth_dir_name = os.path.join(dir_name, '../depth')
        depth_base_name = base_name
        if not os.path.exists(depth_dir_name):
            os.makedirs(depth_dir_name)
        depth_file_path = os.path.join(depth_dir_name, depth_base_name)

        cv2.imwrite(depth_file_path, depth)


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'

    data_roots = [
        '/dataset/vfayezzhang/dataset/SID'
    ]

    mp.spawn(worker, args=(world_size, encoder, model_configs, files), nprocs=world_size, join=True)
