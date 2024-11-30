import os
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models.autoencoder import AutoencoderKL


def load_image(file_path):
    """Load an image from a pickle file and preprocess it."""
    with open(file_path, 'rb') as f:
        image = pickle.load(f)
    
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    image = 2 * image - 1  # Shift to [-1, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return image

if __name__ == "__main__":
    config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/configs/config_local_f8.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model = AutoencoderKL(**config['model']['params'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/weights/no_explosion_10.ckpt")
    image_dir = '/workspace/thomas/latentDiffusion/autoencoderTraining/data/MuaSlices/train'
    image_files = os.listdir(image_dir)

    N = 10000
    #means = torch.zeros(size = (4096, N), device=device)
    flattened_latents = torch.zeros(size = (4096, N), device=device)
    for img_number in range(10):
        image_file = os.path.join(image_dir, image_files[img_number])
        original_image = load_image(image_file)
        if original_image is None:
            continue
        original_image = original_image.to(device)
        
        with torch.no_grad():
            flattened_latent = model.encode(original_image).mean.flatten()
            mean = flattened_latent.mean()
            flattened_latent = flattened_latent - mean
            flattened_latents[:, img_number] = flattened_latent
    with torch.no_grad():
        U, S, Vt = torch.linalg.svd(flattened_latents)
    print("Number of non-zero values:", (S>0).sum().item())
    plt.figure()
    plt.ylabel('Singular Value')
    plt.xlabel('Index')
    plt.plot(S.cpu(), marker = 'o', markersize = 4)
    plt.savefig("svd.png")