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


# 请你把对于每个文件夹，生成他对应的depth文件，dir_name_depth，basename同名，然后把
if __name__ == '__main__':
    data_root = r'../../Dataset/SIDD/test'
    image_paths = []

    for root, dir, files in os.walk(data_root):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(('.jpg', '.png')) and 'depth' not in file:
                image_paths.append(file_path)

    image_paths = sorted(image_paths)
    for image_path in image_paths:
        print(f"Processing {image_path}")
        dir_name = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        depth_dir = dir_name + '_depth'
        os.makedirs(depth_dir, exist_ok=True)
        depth_path = os.path.join(depth_dir, base_name)
        if os.path.exists(depth_path):
            continue
        depth = process_image(image_path, model)
        cv2.imwrite(depth_path, depth)
