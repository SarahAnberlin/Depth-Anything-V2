import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from eva_metrics import delta1_acc_np, delta2_acc_np, delta3_acc_np, abs_relative_difference_np, mse_np
from PIL import Image


def resize_image_with_pil_bilinear(image, new_size):
    pil_image = Image.fromarray(image)  # 从 NumPy 数组创建图像
    resized_image = pil_image.resize(new_size, Image.BILINEAR)  # 使用双线性插值
    return np.array(resized_image)  # 转回 NumPy 数组


def get_center_mask(depth):
    height, width = depth.shape

    # print(f"depth shape: {depth.shape}")

    top = int(0.1 * height)
    bottom = int(0.9 * height)
    left = int(0.1 * width)
    right = int(0.9 * width)

    # 创建一个全 False 的 mask
    center_mask = np.zeros_like(depth, dtype=bool)

    # 将中间区域设为 True
    center_mask[top:bottom, left:right] = True

    return center_mask


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    with open('valid_path.txt', 'r') as f:
        p_paths = f.read().splitlines()

    p_paths = sorted(p_paths)

    cnt = 0
    total_d1 = 0.0
    total_d2 = 0.0
    total_d3 = 0.0
    total_ard = 0.0
    total_mse = 0.0

    print(f"Total number of files: {len(p_paths)}")
    print(f"Top 10 files: {p_paths[:10]}")
    # 创建用于存储每个样本的指标值的列表
    d1_list, d2_list, d3_list = [], [], []
    ard_list, mse_list = [], []

    # 设置保存目录
    save_dir = "scatter_plots"
    os.makedirs(save_dir, exist_ok=True)

    for p_path in p_paths:
        p_depth = cv2.imread(p_path.replace('p_depth', 'p_depth'), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(p_path.replace('p_depth', 'depth'), cv2.IMREAD_GRAYSCALE)

        p_depth = cv2.resize(p_depth, (480, 720), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (480, 720), interpolation=cv2.INTER_LINEAR)

        depth, p_depth = depth.astype(np.float32) / 255.0, p_depth.astype(np.float32) / 255.0

        valid_mask = get_center_mask(depth)
        mask = (depth > 0.002) & (depth < 0.998)
        mask = mask & valid_mask

        if np.sum(mask) == 0:
            print(f"File {p_path} has {np.sum(mask)} valid pixels")
            continue

        cnt += 1
        if cnt % 100 == 0:
            print(f"Progress {cnt}/{len(p_paths)}")

        # 计算各项指标
        d1 = delta1_acc_np(p_depth, depth, mask)
        d2 = delta2_acc_np(p_depth, depth, mask)
        d3 = delta3_acc_np(p_depth, depth, mask)
        ard = abs_relative_difference_np(p_depth, depth, mask)
        mse = mse_np(p_depth, depth, mask)

        # 累加总值
        total_d1 += d1
        total_d2 += d2
        total_d3 += d3
        total_ard += ard
        total_mse += mse

        # 记录每个样本的值
        d1_list.append(d1)
        d2_list.append(d2)
        d3_list.append(d3)
        ard_list.append(ard)
        mse_list.append(mse)

    # 打印平均值
    print(f"avg delta1: {total_d1 / cnt}")
    print(f"avg delta2: {total_d2 / cnt}")
    print(f"avg delta3: {total_d3 / cnt}")
    print(f"avg ard: {total_ard / cnt}")
    print(f"avg mse: {total_mse / cnt}")

    # 创建散点图
    plt.figure(figsize=(12, 8))

    # Delta1
    plt.subplot(2, 3, 1)
    plt.scatter(range(cnt), d1_list, c='r', label='Delta1')
    plt.title('Delta1 per sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Delta1 Accuracy')

    # Delta2
    plt.subplot(2, 3, 2)
    plt.scatter(range(cnt), d2_list, c='g', label='Delta2')
    plt.title('Delta2 per sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Delta2 Accuracy')

    # Delta3
    plt.subplot(2, 3, 3)
    plt.scatter(range(cnt), d3_list, c='b', label='Delta3')
    plt.title('Delta3 per sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Delta3 Accuracy')

    # ARD
    plt.subplot(2, 3, 4)
    plt.scatter(range(cnt), ard_list, c='m', label='ARD')
    plt.title('Absolute Relative Difference per sample')
    plt.xlabel('Sample Index')
    plt.ylabel('ARD')

    # MSE
    plt.subplot(2, 3, 5)
    plt.scatter(range(cnt), mse_list, c='c', label='MSE')
    plt.title('Mean Squared Error per sample')
    plt.xlabel('Sample Index')
    plt.ylabel('MSE')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    output_path = os.path.join(save_dir, "scatter_plots.png")
    plt.savefig(output_path, dpi=300)
    print(f"Scatter plots saved at: {output_path}")
    plt.close()

    # raw_img = cv2.imread('your/image/path')
    # depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
