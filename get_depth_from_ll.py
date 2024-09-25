import matplotlib.pyplot as plt
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
from eva_metrics import abs_relative_difference_np, mse_np, delta2_acc_np, delta1_acc_np, delta3_acc_np


# 读取 .ARW 或 .RAF 文件的函数
def read_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        # 使用 postprocess 方法将 RAW 文件转换为 RGB 图像
        rgb_image = raw.postprocess()
    return rgb_image


def equalize_hist(image_rgb):
    r, g, b = cv2.split(image_rgb)

    # 对每个通道进行直方图均衡
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # 合并均衡后的通道
    equalized_image = cv2.merge((r_eq, g_eq, b_eq))

    return equalized_image


def convert_to_float(image):
    # 将 uint16 图像转换为 float32 图像，范围从 [0, 65535] 转换为 [0.0, 1.0]
    return image.astype(np.float32) / 65535.0


def read_raw_to_rgb(file_path):
    image = read_raw_image(file_path)
    return convert_to_float(image)


def worker(rank, world_size, encoder, model_configs, files):
    # Setup the device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}_{rank}.pth', map_location='cpu'))
    model = model.to(device).eval()

    files_to_process = files[rank::world_size]

    # Process data
    cnt = 0
    print(f'Rank {rank} has {len(files_to_process)} files to process')
    p_depth_path = []
    print(f"top 10 files: {files_to_process[:10]}")
    for file in files_to_process:
        cnt += 1
        if cnt % 100 == 0:
            print(f'Progress {cnt}/{len(files_to_process)}: {file}')
        base_name = os.path.basename(file)
        dir_name = os.path.dirname(file)
        depth_dir_name = os.path.join(dir_name, '../he_ll_p_depth')
        depth_base_name = base_name
        os.makedirs(depth_dir_name, exist_ok=True)
        depth_file_path = os.path.join(depth_dir_name, depth_base_name)
        p_depth_path.append(depth_file_path)
        if os.path.exists(depth_file_path):
            continue
        raw_image = cv2.imread(file)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        raw_image = equalize_hist(raw_image)
        depth = model.infer_image(raw_image)
        cv2.imwrite(depth_file_path, depth)


if __name__ == '__main__':
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'

    files = None
    with open('valid_path.txt', 'r') as f:
        files = f.read().splitlines()

    files = [file.replace('p_depth', 'input') for file in files]

    # world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    world_size = 1
    mp.spawn(worker, args=(world_size, encoder, model_configs, files), nprocs=world_size, join=True)
