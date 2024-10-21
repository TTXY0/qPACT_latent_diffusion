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
    image = (image - image.min()) / (image.max() - image.min())  
    image = 2 * image - 1  # Shift to [-1, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    # print("input image range : ", image.min().item(), image.max().item())
    return image

import matplotlib.pyplot as plt

def show_images(original, reconstructed, save_path=None):
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    original_shape = original_np.shape
    reconstructed_shape = reconstructed_np.shape
    crop_height = (reconstructed_shape[0] - original_shape[0]) // 2
    crop_width = (reconstructed_shape[1] - original_shape[1]) // 2
    cropped_reconstructed_np = reconstructed_np[crop_height:crop_height + original_shape[0],
                                            crop_width:crop_width + original_shape[1]]
    difference_np = np.subtract(original_np, reconstructed_np)
    # print("original range: ", original_np.min(), original_np.max())
    # print("recnstructed range: ", reconstructed_np.min(), reconstructed_np.max())
    vmin = -1
    vmax = 1

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Original Image
    ax[0].imshow(original_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Reconstructed Image
    ax[1].imshow(reconstructed_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")

    # Difference Image
    im = ax[2].imshow(difference_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[2].set_title("Difference")
    ax[2].axis("off")

    # Add a labeled colorbar for the Difference
    cbar = fig.colorbar(im, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()



if __name__ == "__main__":
    # Load the config
    config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/config.yaml'
    #config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/lightning_logs/version_166/checkpoints/epoch=0-step=9999.ckpt'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set up the model
    model = AutoencoderKL(**config['model']['params'])
    #model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/epoch=9-step=33340.ckpt")
    model.init_from_ckpt("/workspace/thomas/latentDiffusion/epoch=6-step=23338.ckpt")
    # model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/kl-f8.ckpt")
    # model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/lightning_logs/version_1/checkpoints/epoch=2-step=20000.ckpt")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    image_dir = '/workspace/thomas/MuaSlices/train' 
    for img_number in range(10) :
        image_file = os.path.join(image_dir, os.listdir(image_dir)[img_number])  # Get the first image file
        original_image = load_image(image_file)


        
        original_image = original_image.to(device)

        with torch.no_grad():
            reconstructed_image, _ = model(original_image)

        #difference_image = torch.abs(original_image - reconstructed_image)

        show_images(original_image, reconstructed_image, save_path=f"6_epochs{img_number}.png")