import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
from depth_anything_v2.dpt import DepthAnythingV2
from eva_metrics import delta1_acc_np, delta2_acc_np, delta3_acc_np, abs_relative_difference_np, \
    threshold_percentage_np, mse_np


def process_image(image_path, model, sigma):
    raw_img = cv2.imread(image_path).astype(np.float32)
    noisy_image = None
    print(f"Raw image dtype: {raw_img.dtype}")
    if sigma != 0:
        noise = np.random.normal(0, sigma, raw_img.shape)
        raw_img = raw_img + noise
    noisy_image = raw_img
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    print(f"Noisy image dtype: {noisy_image.dtype}")
    depth = model.infer_image(noisy_image)  # 生成深度图
    return noisy_image, depth


# 请你把对于每个文件夹，生成他对应的depth文件，dir_name_depth，basename同名，然后把
if __name__ == '__main__':

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        # 'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        # 'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    data_root = r'/dataset/vfayezzhang/dataset/MiniImageNet1k/'
    image_paths = []

    for root, dir, files in os.walk(data_root):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(('.jpg', '.png')) and ('depth' not in file_path):
                image_paths.append(file_path)

    image_paths = sorted(image_paths)

    noise_levels = [0, 15, 25, 50]
    for encoder in model_configs.keys():

        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to(DEVICE).eval()

        for image_path in image_paths:
            print(f"Processing {image_path}")
            noisy_images = []
            depths = []

            for noise_level in noise_levels:
                dir_name = os.path.dirname(image_path)
                base_name = os.path.basename(image_path)
                depth_dir = dir_name + '_depth' + f'_{encoder}_{noise_level}sigma'
                noisy_image_dir = dir_name + '_noisy_image' + f'_{encoder}_{noise_level}sigma'
                os.makedirs(depth_dir, exist_ok=True)
                os.makedirs(noisy_image_dir, exist_ok=True)
                depth_path = os.path.join(depth_dir, base_name)
                noisy_image_path = os.path.join(noisy_image_dir, base_name)

                # Process the image
                noisy_image, depth = process_image(image_path, model, noise_level)

                # Save individual images
                cv2.imwrite(depth_path, depth)
                cv2.imwrite(noisy_image_path, noisy_image)

                # Convert images to RGB for plt (if necessary)
                noisy_image_rgb = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB) if len(
                    noisy_image.shape) == 3 else noisy_image
                depth_rgb = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB) if len(depth.shape) == 3 else depth

                # Add to lists for concatenation
                noisy_images.append(noisy_image_rgb)
                depths.append(depth_rgb)

            # Set up matplotlib figure for concatenation
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows (noisy images, depths), 4 columns (noise levels)

            # Plot noisy images
            for i, noisy_image in enumerate(noisy_images):
                axes[0, i].imshow(noisy_image)
                axes[0, i].axis('off')
                axes[0, i].set_title(f'Noisy Image ({noise_levels[i]} sigma)')

            # Plot depth maps
            for i, depth in enumerate(depths):
                axes[1, i].imshow(depth, cmap='gray')
                axes[1, i].axis('off')
                axes[1, i].set_title(f'Depth Map ({noise_levels[i]} sigma)')

            # Save the concatenated figure
            concat_output_path = os.path.join(dir_name + f'_concat_{encoder}.png')
            plt.tight_layout()
            plt.savefig(concat_output_path)
            plt.close()

            print(f"Concatenated image saved to {concat_output_path}")
