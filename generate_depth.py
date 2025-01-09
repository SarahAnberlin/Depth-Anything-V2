from torchvision.transforms.functional import InterpolationMode
import shutil
import time

import rawpy
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import cv2
import torch
import os
import torch
import torch.multiprocessing as mp
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import cv2
from torchvision.utils import save_image
from dataset.AM2KDataset import AM2KDataset
from dataset.NYUDataset import NYUDataset
from dataset.SintelDataset import SintelDataset
from depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
from torchvision import transforms


def save_fig(image, predict_depth_vis, depth_gt_vis, save_root, id):
    # Create the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot input image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot predicted depth
    axes[1].imshow(predict_depth_vis, cmap="viridis")
    axes[1].set_title("Predicted Depth")
    axes[1].axis("off")

    # Plot ground truth depth
    axes[2].imshow(depth_gt_vis, cmap="viridis")
    axes[2].set_title("Ground Truth Depth")
    axes[2].axis("off")

    # Save and show the figure
    output_path = os.path.join(save_root, f"{id}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to {output_path}")


def save_single_fig(predict_depth, save_root, id):
    # Normalize to range [0, 1]
    predict_depth = (predict_depth - np.min(predict_depth)) / (np.max(predict_depth) - np.min(predict_depth))

    # Apply colormap
    cmap = plt.get_cmap('viridis')  # 使用 'viridis' 颜色映射，可更改为其他映射
    predict_depth_colored = cmap(predict_depth)  # 返回 RGBA 数组
    predict_depth_colored = (predict_depth_colored[:, :, :3] * 255).astype(np.uint8)  # 转换为 RGB

    # Save as PNG using PIL
    save_path = os.path.join(save_root, f"{id}.png")
    print(f"Saving to {save_path}")
    image = Image.fromarray(predict_depth_colored)
    image.save(save_path)


def clip_array_percentile(array, lower_percentile=20, upper_percentile=80):
    """
    Clips the values in the NumPy array to the range defined by the lower and upper percentiles.

    Parameters:
        array (np.ndarray): Input array.
        lower_percentile (float): The lower percentile (default: 20).
        upper_percentile (float): The upper percentile (default: 80).

    Returns:
        np.ndarray: The clipped array.
    """
    # Calculate the percentile values
    lower_bound = np.percentile(array, lower_percentile)
    upper_bound = np.percentile(array, upper_percentile)

    # Clip the array
    clipped_array = np.clip(array, lower_bound, upper_bound)

    return clipped_array


def worker(rank, world_size, encoder, model_configs, dataset, save_root):
    # Setup the device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    # Load the model
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(
        torch.load(f'/dataset/vfayezzhang/test/DIR/main/checkpoints/depth_anything_v2_{encoder}_{rank}.pth',
                   map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for idx, data in enumerate(dataset):
            next_save_path = os.path.join(save_root, f"{idx + 2}.png")
            if os.path.exists(next_save_path):
                continue

            image = data
            print(f"Image shape: {image.shape}")
            image = image.unsqueeze(0).to(device)
            h, w = image.shape[-2:]
            pad_h = 14 - ((h + 14) % 14)
            pad_w = 14 - ((w + 14) % 14)
            new_h = h + pad_h
            new_w = w + pad_w
            image_test = transforms.Resize((new_h, new_w), interpolation=InterpolationMode.BICUBIC)(image)
            image_test = transform(image_test)
            prediction = model(image_test, test=True)

            prediction = transforms.Resize((h, w), interpolation=InterpolationMode.BILINEAR)(prediction)
            prediction = (prediction - torch.min(prediction)) / (torch.max(prediction) - torch.min(prediction))
            save_path = os.path.join(save_root, f"{idx + 1}.png")
            save_image(prediction, save_path)


def get_dataset(dataset_name):
    # if dataset_name == "Hypersim":
    #     return HypersimDataset()
    if dataset_name == "Sintel":
        return SintelDataset()
    if dataset_name == 'NYUv2':
        return NYUDataset()
    if dataset_name == 'AM2K':
        return AM2KDataset()


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    # dataset_name = 'Sintel'
    # dataset_name = 'Sintel'
    # dataset_name = "NYUv2"
    dataset_name = 'AM2K'
    dataset = get_dataset(dataset_name)
    save_root = '/dataset/vfayezzhang/test/depth-pro/infer/vis/dav2-test/'
    save_root = os.path.join(save_root, dataset_name)
    os.makedirs(save_root, exist_ok=True)

    print(f"Length of dataset: {len(dataset)}")
    encoder = 'vitl'

    world_size = 1
    print(f'World size: {world_size}')

    mp.spawn(worker, args=(world_size, encoder, model_configs, dataset, save_root), nprocs=world_size,
             join=True)
