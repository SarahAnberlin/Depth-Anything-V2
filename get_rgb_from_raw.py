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


# 读取 .ARW 或 .RAF 文件的函数
def read_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        print(f'raw_image.shape: {raw.raw_image.shape}')  # H W
        print(f'raw_image.dtype: {raw.raw_image.dtype}')  # uint16
        print(f'raw_image.max: {raw.raw_image.max()} min: {raw.raw_image.min()}')
        # 使用 postprocess 方法将 RAW 文件转换为 RGB 图像
        rgb_image = raw.postprocess()
        print(f'rgb_image.shape: {rgb_image.shape}')  # H W C
        print(f'rgb_image.dtype: {rgb_image.dtype}')
        print(f'rgb_image.max: {rgb_image.max()} min: {rgb_image.min()}')
    return rgb_image




def read_raw_to_rgb(file_path):
    image = read_raw_image(file_path)
    return image


def gather_files(data_roots):
    all_files = []
    for data_root in data_roots:
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.ARW') or file.endswith('.RAF'):
                    all_files.append(os.path.join(root, file))
    return all_files


def worker(rank, world_size, encoder, model_configs, files):
    # Setup the device
    # device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    #
    # # Load the model
    # model = DepthAnythingV2(**model_configs[encoder])
    # model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}_{rank}.pth', map_location='cpu'))
    # model = model.to(device).eval()

    files_to_process = files[rank::world_size]

    # Process data
    for file in files_to_process:
        rgb_image = read_raw_to_rgb(file)
        if 'short' not in file:
            continue
        # depth = model.infer_image(raw_image)
        # base_name = os.path.basename(file)
        # dir_name = os.path.dirname(file)
        # depth_dir_name = os.path.join(dir_name, '../depth')
        # if not os.path.exists(depth_dir_name):
        #     os.makedirs(depth_dir_name)
        # depth_base_name = base_name.replace('.ARW', '.png').replace('.RAF', '.png')
        # depth_file_path = os.path.join(depth_dir_name, depth_base_name)

        # cv2.imwrite(depth_file_path, depth)
        print(f'raw_image.shape: {rgb_image.shape}')  # H W C
        # Save the raw image
        base_name = os.path.basename(file).replace('.ARW', '.png').replace('.RAF', '.png')
        dir_name = os.path.dirname(file).replace('long', 'long_rgb').replace('short', 'short_rgb')
        os.makedirs(dir_name, exist_ok=True)
        rgb_path = os.path.join(dir_name, base_name)
        cv2.imwrite(rgb_path, rgb_image)


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

    files = gather_files(data_roots)
    files = sorted(files)
    world_size = 8
    print(f'World size: {world_size}')
    # source_file = f'checkpoints/depth_anything_v2_{encoder}.pth'
    # destination_dir = 'checkpoints'
    #
    # for rank in range(world_size):
    #     destination_file = f'{destination_dir}/depth_anything_v2_{encoder}_{rank}.pth'
    #     if not os.path.exists(destination_file):
    #         shutil.copy(source_file, destination_file)
    #         print(f'Copied to {destination_file}')

    mp.spawn(worker, args=(world_size, encoder, model_configs, files), nprocs=world_size, join=True)
