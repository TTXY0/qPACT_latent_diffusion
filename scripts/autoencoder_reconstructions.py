import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt


from models.autoencoder_downsample import AutoencoderKL
config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/configs/config_local_f8_noExplosion.yaml'
model_path = "/workspace/thomas/latentDiffusion/autoencoderTraining/weights/kl-f8-10.ckpt"

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


def show_images(original, reconstructions, save_path=None):
    """Display all images on a single grid layout."""
    original_np = original.squeeze(0).squeeze(0).cpu().numpy()
    #untrained_np = untrained_reconstructed.squeeze(0).squeeze(0).cpu().numpy()
    vmin, vmax = -1, 1  # Value range for display

    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    for col in range(5):
        for row in range(5):
            print(col * 5 + row )
            image = reconstructions[col * 5 + row ].squeeze().cpu().numpy()
            im = ax[col, row].imshow(image, cmap='viridis', vmin=vmin, vmax=vmax)
            ax[col, row].axis("off")
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the model
    model = AutoencoderKL(**config['model']['params'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    image_dir = '/workspace/thomas/latentDiffusion/autoencoderTraining/data/MuaSlices/train'
    image_files = os.listdir(image_dir)
    reconstructions = []
    index = 0
    for img_number in range(25):
        image_file = os.path.join(image_dir, image_files[img_number])
        original_image = load_image(image_file)
        if original_image is None:
            continue
        original_image = original_image.to(device)
        

        model.init_from_ckpt(model_path)
        with torch.no_grad():
            reconstruction, _ = model(original_image)
        reconstructions.append(reconstruction)
        index += 1
    show_images(original_image, reconstructions, save_path=f"/workspace/thomas/latentDiffusion/autoencoderTraining/scripts/imgs/autoencoder_recon_{img_number}_svd.png")
