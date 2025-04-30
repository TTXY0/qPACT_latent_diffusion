import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models.autoencoder16x16x4 import AutoencoderKL
from PIL import Image


def load_image(file_path):
    with open(file_path, 'rb') as f:
        image = pickle.load(f)
    
    image = image.numpy()
    image = (image - image.min()) / (image.max() - image.min())  
    image = 2 * image - 1  # Shift to [-1, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    return image

def show_images(original, no_training_reconstruction, train_reconstruction, save_path=None):
    original_np = original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    untrained_np = no_training_reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy()
    trained_np = train_reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy()

    vmin, vmax = -1, 1

    fig, ax = plt.subplots(1, 3, figsize=(24, 6))


    ax[0].imshow(original_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title("Original Image")
    ax[0].axis("off")



    ax[1].imshow(trained_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title("Trained")
    ax[1].axis("off")

    difference_np = np.subtract(original_np, trained_np)
    im = ax[2].imshow(difference_np, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[2].set_title("Difference (Trained - Original)")
    ax[2].axis("off")

    cbar = fig.colorbar(im, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Difference Intensity')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":

    config_path = '/home/twynn/Desktop/latentDiffusion/autoencoderTraining/configs/config_bevo_16x16x4_untrained.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model_no_training = AutoencoderKL(**config['model']['params'])
    model_no_training.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_no_training = model_no_training.to(device)

    config_path = '/home/twynn/Desktop/latentDiffusion/autoencoderTraining/configs/config_bevo_16x16x4_trained.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model_trained = AutoencoderKL(**config['model']['params'])
    model_trained.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_trained = model_trained.to(device)

    image_dir = '/home/twynn/Desktop/latentDiffusion/autoencoderTraining/data/MuaSlices/train'
    for img_number in range(10):
        image_file = os.path.join(image_dir, os.listdir(image_dir)[img_number])
        original_image = load_image(image_file).to(device)
        

        with torch.no_grad():
            no_training_reconstruction, _ = model_no_training(original_image)

        with torch.no_grad():
            trained_reconstruction, _ = model_trained(original_image)

        show_images(original_image, no_training_reconstruction, trained_reconstruction, save_path=f"comparisonl_{img_number}.png")
