import cv2
import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2
from eva_metrics import delta1_acc_np, delta2_acc_np, delta3_acc_np, abs_relative_difference_np, \
    threshold_percentage_np, mse_np

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
    dir_name = 'src/image/'
    noise_types = ['gaussian', 'poisson', 'salt_and_pepper']
    files = [f for f in os.listdir(dir_name) if f.endswith(('.jpg', '.png')) and 'depth' not in f]
    files = sorted(files)
    image_paths = [os.path.join(dir_name, f) for f in files]
    # Generate depth images for each image
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        depth = process_image(image_path, model)
        depth_dir = os.path.join(dir_name, '../depth')
        os.makedirs(depth_dir, exist_ok=True)
        depth_path = os.path.join(depth_dir, base_name)
        if not os.path.exists(depth_path):
            cv2.imwrite(depth_path, depth)

    # Generate depth images for degraded images
    for noise_type in noise_types:

        low_light_dir = os.path.join(dir_name, '../low_light')
        restore_image_dir = os.path.join(dir_name, f'../restore_image_{noise_type}')
        low_light_noisy_dir = os.path.join(dir_name, f'../low_light_noisy_{noise_type}')

        low_light_depth_dir = os.path.join(dir_name, '../low_light_depth')
        restore_image_depth_dir = os.path.join(dir_name, f'../restore_image_{noise_type}_depth')
        low_light_noisy_depth_dir = os.path.join(dir_name, f'../low_light_noisy_{noise_type}_depth')

        os.makedirs(low_light_depth_dir, exist_ok=True)
        os.makedirs(restore_image_depth_dir, exist_ok=True)
        os.makedirs(low_light_noisy_depth_dir, exist_ok=True)

        for image_path in image_paths:
            base_name = os.path.basename(image_path)
            low_light_image_path = os.path.join(low_light_dir, base_name)
            restore_image_path = os.path.join(restore_image_dir, base_name)
            low_light_noisy_image_path = os.path.join(low_light_noisy_dir, base_name)

            low_light_depth = process_image(low_light_image_path, model)
            restore_image_depth = process_image(restore_image_path, model)
            low_light_noisy_depth = process_image(low_light_noisy_image_path, model)

            low_light_depth_path = os.path.join(low_light_depth_dir, base_name)
            restore_image_depth_path = os.path.join(restore_image_depth_dir, base_name)
            low_light_noisy_depth_path = os.path.join(low_light_noisy_depth_dir, base_name)
            if not os.path.exists(low_light_depth_path):
                cv2.imwrite(low_light_depth_path, low_light_depth)
            if not os.path.exists(restore_image_depth_path):
                cv2.imwrite(restore_image_depth_path, restore_image_depth)
            if not os.path.exists(low_light_noisy_depth_path):
                cv2.imwrite(low_light_noisy_depth_path, low_light_noisy_depth)
