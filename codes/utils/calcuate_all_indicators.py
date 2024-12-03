import numpy as np
from PIL import Image
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import shannon_entropy
from tabulate import tabulate
import os

def rmse(image1, image2):
    return np.sqrt(((image1 - image2) ** 2).mean())

def sam(image1, image2):
    image1 = image1.reshape(-1, 3)
    image2 = image2.reshape(-1, 3)
    cos_theta = np.clip(
        np.sum(image1 * image2, axis=1) / (np.linalg.norm(image1, axis=1) * np.linalg.norm(image2, axis=1)), -1, 1)
    return np.mean(np.arccos(cos_theta)) * 180 / np.pi

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image) / 255.0  # 归一化

def calculate_lpips(image1, image2, device='cpu'):
    model = lpips.LPIPS(net='alex').to(device)
    image1 = torch.from_numpy(np.transpose(image1, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    image2 = torch.from_numpy(np.transpose(image2, (2, 0, 1)).astype(np.float32)).unsqueeze(0).to(device)
    return model(image1, image2).item()

def calculate_pi(image1, image2):
    # 计算图像熵差异
    entropy1 = shannon_entropy(image1)
    entropy2 = shannon_entropy(image2)
    return np.abs(entropy1 - entropy2)

def calculate_metrics(image1_path, image2_path, device='cpu'):
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    lpips_value = calculate_lpips(image1, image2, device)
    pi_value = calculate_pi(image1, image2)
    psnr_value = psnr(image1, image2)
    ssim_value = ssim(image1, image2, multichannel=True)
    rmse_value = rmse(image1, image2)
    sam_value = sam(image1, image2)
    return {
        "LPIPS": lpips_value,
        "PI": pi_value,
        "PSNR": psnr_value,
        "SSIM": ssim_value,
        "RMSE": rmse_value,
        "SAM": sam_value
    }

def process_folder(input_folder, reference_image_path, device='cpu'):
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            results = calculate_metrics(reference_image_path, file_path, device)
            table_data = [(metric, value) for metric, value in results.items()]
            print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))


input_folder = "D:/PycharmProjects/ruanzhu/data_preprocessing/dui_HR.jpg"
reference_image_path = "data_preprocessing/STGAN.png"
process_folder(input_folder, reference_image_path)
