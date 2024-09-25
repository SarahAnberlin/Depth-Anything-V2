import argparse
import json

import cv2
import numpy as np
import torch
import os

from PIL import Image
from matplotlib import pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2
from eva_metrics import delta1_acc_np, delta2_acc_np, delta3_acc_np, abs_relative_difference_np, \
    threshold_percentage_np, mse_np


def process_image(image_path, model):
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img)  # 生成深度图
    return depth


def generate_depth(image_path, model, return_depth=True):
    # print(f"Processing {image_path}")
    base_name = os.path.basename(image_path)
    depth_dir_name = os.path.dirname(image_path) + '_depth'
    depth_path = os.path.join(depth_dir_name, base_name)
    depth = None
    if not os.path.exists(depth_path):
        depth = process_image(image_path, model)
        os.makedirs(depth_dir_name, exist_ok=True)
        print(f"Saving depth to {depth_path}")
        cv2.imwrite(depth_path, depth)
        return depth
    elif return_depth:
        depth = cv2.imread(depth_path)
        return depth
    return None


def save_comparison_plot(rgb_path, ll_path, rgb_depth, ll_depth):
    # 加载原始RGB图像和LL图像
    rgb_image = Image.open(rgb_path)
    ll_image = Image.open(ll_path)

    # 获取路径名并建立summary文件夹
    dirname = os.path.dirname(rgb_path)
    summary_dir = os.path.join(dirname, "../summary")
    os.makedirs(summary_dir, exist_ok=True)

    # 获取文件名
    filename = os.path.basename(ll_path)
    filename_without_ext = os.path.splitext(filename)[0]

    # 绘制对比图
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # 设置每个图像，并增加边框
    axs[0, 0].imshow(rgb_image)
    axs[0, 0].set_title('RGB Image', fontsize=16)
    axs[0, 0].patch.set_edgecolor('black')
    axs[0, 0].patch.set_linewidth(2)

    axs[0, 1].imshow(ll_image)
    axs[0, 1].set_title('LL Image', fontsize=16)
    axs[0, 1].patch.set_edgecolor('black')
    axs[0, 1].patch.set_linewidth(2)

    axs[1, 0].imshow(rgb_depth, cmap='plasma')
    axs[1, 0].set_title('RGB Depth', fontsize=16)
    axs[1, 0].patch.set_edgecolor('black')
    axs[1, 0].patch.set_linewidth(2)

    axs[1, 1].imshow(ll_depth, cmap='plasma')
    axs[1, 1].set_title('LL Depth', fontsize=16)
    axs[1, 1].patch.set_edgecolor('black')
    axs[1, 1].patch.set_linewidth(2)

    # 移除多余的坐标轴
    for ax in axs.flat:
        ax.axis('off')

    # 减少图片之间的间隙
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # 保存图像到 summary 文件夹中
    save_path = os.path.join(summary_dir, f"{filename_without_ext}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


# 请你把对于每个文件夹，生成他对应的depth文件，dir_name_depth，basename同名，然后把
if __name__ == '__main__':

    cnt = 0
    abs_rel_diff_list = []
    mse_list = []
    delta1_list = []
    delta2_list = []
    delta3_list = []

    parser = argparse.ArgumentParser(description='Generate depth images for SID dataset')
    parser.add_argument('--reverse', action='store_true', help='Reverse the order of the images')
    args = parser.parse_args()

    reverse = args.reverse

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))

    with open('meta_data.json', 'r') as f:
        meta_data = json.load(f)
    meta_data = sorted(meta_data, key=lambda x: x['rgb_path'], reverse=reverse)

    #
    for data in meta_data:

        rgb_path = data['rgb_path']
        ll_path = data['ll_path']

        rgb_depth = generate_depth(rgb_path, model)
        ll_depth = generate_depth(ll_path, model)

        print(f"rgb_depth.shape: {rgb_depth.shape}")
        print(f"ll_depth.shape: {ll_depth.shape}")

        rgb_depth_np = rgb_depth.transpose(2, 0, 1).astype(np.float32) / 255.0
        ll_depth_np = ll_depth.transpose(2, 0, 1).astype(np.float32) / 255.0

        abs_rel_diff = abs_relative_difference_np(rgb_depth_np, ll_depth_np)
        mse_val = mse_np(rgb_depth_np, ll_depth_np)
        delta1_acc = delta1_acc_np(rgb_depth_np, ll_depth_np)
        delta2_acc = delta2_acc_np(rgb_depth_np, ll_depth_np)
        delta3_acc = delta3_acc_np(rgb_depth_np, ll_depth_np)

        abs_rel_diff_list.append(abs_rel_diff)
        mse_list.append(mse_val)
        delta1_list.append(delta1_acc)
        delta2_list.append(delta2_acc)
        delta3_list.append(delta3_acc)

        rgb_depth = rgb_depth.transpose(1, 2, 0)
        ll_depth = ll_depth.transpose(1, 2, 0)
        save_comparison_plot(rgb_path, ll_path, rgb_depth, ll_depth)

        cnt += 1
        if cnt % 100 == 0:
            print(f"Processed {cnt} images")

    # 计算所有图像的平均指标
    avg_abs_rel_diff = np.mean(abs_rel_diff_list)
    avg_mse = np.mean(mse_list)
    avg_delta1 = np.mean(delta1_list)
    avg_delta2 = np.mean(delta2_list)
    avg_delta3 = np.mean(delta3_list)

    print(f"Average Absolute Relative Difference: {avg_abs_rel_diff}")
    print(f"Average MSE: {avg_mse}")
    print(f"Average Delta1: {avg_delta1}")
    print(f"Average Delta2: {avg_delta2}")
    print(f"Average Delta3: {avg_delta3}")
