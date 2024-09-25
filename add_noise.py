import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import random

import random
import numpy as np
from PIL import Image, ImageEnhance
import os

from matplotlib import pyplot as plt

import random
import numpy as np
from PIL import Image, ImageEnhance


def add_gaussian_noise(image_array, noise_std):
    """添加高斯噪声"""
    noise = np.random.normal(0, noise_std, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 1)
    return noisy_image


def add_poisson_noise(image_array):
    """添加泊松噪声"""
    noisy_image = np.random.poisson(image_array * 255) / 255.0
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def add_salt_and_pepper_noise(image_array, amount=0.1, salt_vs_pepper=0.5):
    """添加椒盐噪声"""
    noisy_image = np.copy(image_array)
    num_salt = np.ceil(amount * image_array.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image_array.size * (1.0 - salt_vs_pepper))

    # 添加盐噪声 (白色像素)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
    noisy_image[tuple(coords)] = 1

    # 添加椒噪声 (黑色像素)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
    noisy_image[tuple(coords)] = 0

    return noisy_image


def simulate_low_light(image_path, brightness_range=(0.2, 0.21), noise_range=(0.1, 0.11), noise_type='gaussian'):
    """
    调低图像亮度并添加高斯噪声以模拟低光照环境，然后恢复亮度以便更容易看到噪声。

    参数:
    - image_path: str, 图像文件路径
    - brightness_range: tuple, 调整亮度的范围 (默认 (0.1, 0.11))
    - noise_range: tuple, 高斯噪声标准差范围 (默认 (0.49, 0.5))

    返回:
    - 低光照加噪声并恢复亮度后的 PIL 图像
    """

    # 打开图像
    image = Image.open(image_path)
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)

    # 在给定范围内随机生成亮度因子
    brightness_factor = random.uniform(*brightness_range)
    enhancer = ImageEnhance.Brightness(image)
    low_light_image = enhancer.enhance(brightness_factor)

    # 将图像转换为 numpy 数组并归一化到 [0, 1]
    image_array = np.asarray(low_light_image) / 255.0

    # 随机选择噪声类型：高斯噪声、泊松噪声或椒盐噪声
    if noise_type is None:
        noise_type = random.choice(['gaussian', 'poisson', 'salt_and_pepper'])

    noisy_image = None
    if noise_type == 'gaussian':
        noise_std = random.uniform(*noise_range)
        noisy_image = add_gaussian_noise(image_array, noise_std)
    elif noise_type == 'poisson':
        noisy_image = add_poisson_noise(image_array)
    elif noise_type == 'salt_and_pepper':
        noisy_image = add_salt_and_pepper_noise(image_array)

    low_light_dir = os.path.join(dir_name, f'../low_light')
    restore_image_dir = os.path.join(dir_name, f'../restore_image_{noise_type}')
    low_light_noisy_dir = os.path.join(dir_name, f'../low_light_noisy_{noise_type}')

    os.makedirs(low_light_dir, exist_ok=True)
    os.makedirs(restore_image_dir, exist_ok=True)
    os.makedirs(low_light_noisy_dir, exist_ok=True)

    low_light_noisy_image = Image.fromarray((noisy_image * 255).astype(np.uint8))
    # 将噪声图像恢复到原始亮度
    restored_image = noisy_image * (1 / brightness_factor)
    restored_image = np.clip(restored_image, 0, 1)  # 确保像素值在 [0, 1] 之间

    # 将恢复亮度的数组转换回 PIL 图像
    restored_image_pil = Image.fromarray((restored_image * 255).astype(np.uint8))

    # Save the images
    low_light_image.save(os.path.join(low_light_dir, base_name))
    low_light_noisy_image.save(os.path.join(low_light_noisy_dir, base_name))
    restored_image_pil.save(os.path.join(restore_image_dir, base_name))


if __name__ == '__main__':
    # 设置随机种子以便复现结果
    random.seed(0)
    np.random.seed(0)
    image_root = 'src/image'
    image_files = [os.path.join(image_root, file) for file in os.listdir(image_root) if
                   (file.endswith('.jpg') or file.endswith('.png')) and 'depth' not in file]
    noise_types = ['gaussian', 'poisson', 'salt_and_pepper']
    for image_file in image_files:
        for noise_type in noise_types:
            simulate_low_light(image_file, noise_type=noise_type)
