import os
import sys
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import AutoencoderKL
from PIL import Image

# Load and preprocess a single image
def load_image(file_path):
    with open(file_path, 'rb') as f:
        image = pickle.load(f)
    
    image = image.numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    image = 2 * image - 1  # Shift to [-1, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add channel and batch dimension
    return image

def show_images(original, reconstructed, difference, save_path=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")

    ax[2].imshow(difference.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    ax[2].set_title("Difference")
    ax[2].axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Load the config
    config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set up the model
    model = AutoencoderKL(**config['model']['params'])
    model.init_from_ckpt(config['model']['params']['ckpt_path'])
    model.eval()

    # Load a single image
    image_dir = '/workspace/thomas/MuaSlices/train' 
    image_file = os.path.join(image_dir, os.listdir(image_dir)[0])  # Get the first image file
    original_image = load_image(image_file)

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    original_image = original_image.to(device)

    with torch.no_grad():
        reconstructed_image, _ = model(original_image)

    difference_image = torch.abs(original_image - reconstructed_image)

    show_images(original_image, reconstructed_image, difference_image, save_path="reconstruction_comparison.png")