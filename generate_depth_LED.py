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


def convert_to_float(image):
    # 将 uint16 图像转换为 float32 图像，范围从 [0, 65535] 转换为 [0.0, 1.0]
    return image.astype(np.float32) / 65535.0


def read_raw_to_rgb(file_path):
    image = read_raw_image(file_path)
    return convert_to_float(image)


def gather_files(data_roots):
    all_files = []
    for data_root in data_roots:
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    all_files.append(os.path.join(root, file))
    return all_files


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
        depth_dir_name = os.path.join(dir_name, '../p_depth')
        depth_base_name = base_name
        os.makedirs(depth_dir_name, exist_ok=True)
        depth_file_path = os.path.join(depth_dir_name, depth_base_name)
        p_depth_path.append(depth_file_path)
        if os.path.exists(depth_file_path):
            continue
        raw_image = cv2.imread(file)
        depth = model.infer_image(raw_image)

        cv2.imwrite(depth_file_path, depth)

    # Caculate the error between gt and plot the distribution
    error = []
    threshold = 0.11
    valid_path = []
    for file in p_depth_path:
        gt_file = file.replace('p_depth', 'depth')
        if not os.path.exists(gt_file):
            continue
        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        gt = gt.astype(np.float32) / 255.0
        pred = pred.astype(np.float32) / 255.0
        # print(f"Gt shape: {gt.shape}, Pred shape: {pred.shape}")
        mask = (gt > 0.002) & (gt < 0.998)
        mse = mse_np(pred, gt, mask)
        # print(f"Gt max: {gt.max()}, Gt min: {gt.min()}")
        # print(f"Pred max: {pred.max()}, Pred min: {pred.min()}")
        absrel = abs_relative_difference_np(pred, gt, mask)
        if mse <= threshold and absrel < 10 and np.sum(mask) >= (0.8 * mask.shape[-1] * mask.shape[-2]):
            valid_path.append(file)

        error.append(mse)
    print(f"Total kept files: {len(valid_path)}")
    with open('valid_path.txt', 'w') as f:
        for path in valid_path:
            f.write(path + '\n')

    error = np.array(error)
    minn = error.min()
    maxx = error.max()
    print(f"mean: {error.mean()}, std: {error.std()}, min: {minn}, max: {maxx}")

    Q1 = np.percentile(error, 25)  # 第一四分位数
    Q2 = np.percentile(error, 50)  # 中位数
    Q3 = np.percentile(error, 75)  # 第三四分位数
    _90 = np.percentile(error, 90)

    print(f"第一四分位数 (Q1): {Q1}")
    print(f"中位数 (Q2): {Q2}")
    print(f"第三四分位数 (Q3): {Q3}")
    print(f"90%分位数: {_90}")

    num_bins = 100

    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.hist(error, bins=num_bins, color='blue', alpha=0.7, range=(minn, maxx))
    plt.title('MSE Distribution')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.ylabel('Frequency')

    # Save the histogram as an image
    output_path = 'error_distribution.png'  # Change this to your desired output path
    plt.savefig(output_path)

    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(error)), error, color='blue', alpha=0.7, marker='o')
    plt.title('MSE Distribution')
    plt.xlabel('Sample Index')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.axhline(y=np.mean(error), color='red', linestyle='--', label='Mean MSE')  # 可选，标记均值
    plt.legend()

    # 保存图像
    plt.savefig('scatter.png', dpi=300)  # 可自定义文件名和分辨率
    plt.close()  # 关闭图像以释放内存


if __name__ == '__main__':
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'

    data_roots = [
        '/dataset/vfayezzhang/dataset/LED/train/target',
        '/dataset/vfayezzhang/dataset/LED/val/target',
    ]

    files = gather_files(data_roots)
    # world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    world_size = 1
    mp.spawn(worker, args=(world_size, encoder, model_configs, files), nprocs=world_size, join=True)
