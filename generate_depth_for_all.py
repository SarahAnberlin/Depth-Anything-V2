import cv2
import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2
from eva_metrics import delta1_acc_np, delta2_acc_np, delta3_acc_np, abs_relative_difference_np, \
    threshold_percentage_np, mse_np


def process_image(image_path, model):
    raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img)  # 生成深度图
    return depth


def process_folder(image_dir, model, gt_depth, noise_type):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png')) and 'depth' not in f]
    metrics = {}

    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        noisy_depth = process_image(img_path, model)

        # 计算误差
        delta1 = delta1_acc_np(noisy_depth, gt_depth)
        delta2 = delta2_acc_np(noisy_depth, gt_depth)
        delta3 = delta3_acc_np(noisy_depth, gt_depth)
        abs_rel_diff = abs_relative_difference_np(noisy_depth, gt_depth)
        mse = mse_np(noisy_depth, gt_depth)

        # 记录误差
        metrics[image_file] = {
            'delta1': delta1,
            'delta2': delta2,
            'delta3': delta3,
            'abs_rel_diff': abs_rel_diff,
            'mse': mse
        }

        # 保存深度图
        depth_path = os.path.join(image_dir, f'{os.path.splitext(image_file)[0]}_depth_{noise_type}.png')
        cv2.imwrite(depth_path, noisy_depth)

    return metrics


# 请你把对于每个文件夹，生成他对应的depth文件，dir_name_depth，basename同名，然后把
if __name__ == '__main__':

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

    data_root = "/dataset/vfayezzhang/dataset/LOLdataset/eval15"
    for root, dir, files in os.walk(data_root):
        for file in files:
            if 'depth' in root or 'depth' in file or not (file.endswith('.jpg') or file.endswith('.png')):
                continue
            image_path = os.path.join(root, file)
            print(f"Processing {image_path}")
            depth = process_image(image_path, model)
            depth_dir_name = os.path.join(root, f'{dir}_depth')
            os.makedirs(depth_dir_name, exist_ok=True)
            depth_path = os.path.join(depth_dir_name, file)
            if not os.path.exists(depth_path):
                cv2.imwrite(depth_path, depth)
