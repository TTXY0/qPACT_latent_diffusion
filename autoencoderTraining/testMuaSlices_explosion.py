import os
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models.autoencoder_no_explosion import AutoencoderKL as no_explosion
from models.autoencoder_explosion import AutoencoderKL as explosion
from PIL import Image


def load_image(file_path):
    with open(file_path, 'rb') as f:
        image = pickle.load(f)
    
    image = image.numpy()
    image = (image - image.min()) / (image.max() - image.min())  
    image = 2 * image - 1  # Shift to [-1, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    return image

def show_images(original, no_explosion_reconstructed, explosion_reconstructed, save_path=None):
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    no_explosion_np = no_explosion_reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    explosion_np = explosion_reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()

    vmin, vmax = -1, 1

    fig, ax = plt.subplots(1, 5, figsize=(24, 6))


    ax[0].imshow(original_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title("Original Image")
    ax[0].axis("off")


    ax[1].imshow(no_explosion_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title("No Explosion")
    ax[1].axis("off")


    ax[2].imshow(explosion_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[2].set_title("Explosion")
    ax[2].axis("off")


    difference_np = np.subtract(original_np, no_explosion_np)
    im = ax[3].imshow(difference_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[3].set_title("Difference (No Explosion - Original)")
    ax[3].axis("off")

    difference_np = np.subtract(original_np, explosion_np)
    im = ax[4].imshow(difference_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[4].set_title("Difference (Explosion - Original)")
    ax[4].axis("off")

    cbar = fig.colorbar(im, ax=ax[3], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    cbar = fig.colorbar(im, ax=ax[4], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":

    config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/configs/config_local_f8_noExplosion.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model_no_explosion = no_explosion(**config['model']['params'])
    model_no_explosion.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_no_explosion = model_no_explosion.to(device)

    config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/configs/config_local_f8_explosion.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model_explosion = explosion(**config['model']['params'])
    model_explosion.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_explosion = model_explosion.to(device)

    image_dir = '/workspace/thomas/latentDiffusion/autoencoderTraining/data/MuaSlices/train'
    for img_number in range(10):
        image_file = os.path.join(image_dir, os.listdir(image_dir)[img_number])
        original_image = load_image(image_file).to(device)
        

        with torch.no_grad():
            no_explosion_reconstructed, _ = model_no_explosion(original_image)

        with torch.no_grad():
            explosion_reconstructed, _ = model_explosion(original_image)

        show_images(original_image, no_explosion_reconstructed, explosion_reconstructed, save_path=f"comparison_{img_number}.png")
