import os
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import AutoencoderKL
from PIL import Image


def load_image(file_path):
    with open(file_path, 'rb') as f:
        image = pickle.load(f)
    
    image = image.numpy()
    image = (image - image.min()) / (image.max() - image.min())  
    image = 2 * image - 1  # Shift to [-1, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    return image

def show_images(original, untrained_reconstructed, trained_reconstructed, save_path=None):
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    untrained_np = untrained_reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    trained_np = trained_reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()

    vmin, vmax = -1, 1

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))


    ax[0].imshow(original_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title("Original Image")
    ax[0].axis("off")


    ax[1].imshow(untrained_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title("Untrained Reconstruction")
    ax[1].axis("off")


    ax[2].imshow(trained_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[2].set_title("Trained Reconstruction")
    ax[2].axis("off")


    difference_np = np.subtract(original_np, trained_np)
    im = ax[3].imshow(difference_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[3].set_title("Difference (Trained - Original)")
    ax[3].axis("off")

    cbar = fig.colorbar(im, ax=ax[3], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":

    config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)


    model = AutoencoderKL(**config['model']['params'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    image_dir = '/workspace/thomas/MuaSlices/train'
    for img_number in range(10):
        image_file = os.path.join(image_dir, os.listdir(image_dir)[img_number])
        original_image = load_image(image_file).to(device)
        
        model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/kl-f8.ckpt")

        with torch.no_grad():
            untrained_reconstructed, _ = model(original_image)

        model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/tb_logs/autoencoder/SSIM_LOSS/checkpoints/epoch=6-step=46668.ckpt")

        with torch.no_grad():
            trained_reconstructed, _ = model(original_image)

        show_images(original_image, untrained_reconstructed, trained_reconstructed, save_path=f"comparison_{img_number}.png")
