import os
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models.autoencoder_no_explosion import AutoencoderKL as no_explosion


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


def show_images(original, untrained_reconstructed, trained_reconstructed, save_path=None):
    """Display all images on a single grid layout."""
    # Convert tensors to NumPy arrays
    original_np = original.squeeze(0).squeeze(0).cpu().numpy()
    untrained_np = untrained_reconstructed.squeeze(0).squeeze(0).cpu().numpy()
    trained_np = trained_reconstructed.squeeze(0).squeeze(0).cpu().numpy()
    difference_np = trained_np - original_np

    # Combine all images into a single grid
    images = [original_np, untrained_np, trained_np, difference_np]
    titles = ["Original Image", "Untrained Reconstruction", "Trained Reconstruction", "Difference"]
    vmin, vmax = -1, 1  # Value range for display

    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Concatenate images horizontally
    combined_image = np.concatenate(
        [((img - vmin) / (vmax - vmin)) for img in images], axis=1
    )  # Normalize images for consistent display

    # Display the combined image
    ax.imshow(combined_image, cmap='viridis')
    ax.axis("off")

    # Set titles as text on the top of the grid
    x_positions = np.linspace(0, combined_image.shape[1], len(titles), endpoint=False)
    for idx, title in enumerate(titles):
        ax.text(x_positions[idx] + combined_image.shape[1] / len(titles) / 2, -10, title, ha='center', fontsize=12)

    # Save or show the figure
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":

    # Load the model configuration
    config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/configs/config_local_f8_noExplosion.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the model
    model = no_explosion(**config['model']['params'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define the image directory
    image_dir = '/workspace/thomas/latentDiffusion/autoencoderTraining/data/MuaSlices/train'
    image_files = os.listdir(image_dir)

    for img_number in range(10):
        image_file = os.path.join(image_dir, image_files[img_number])
        original_image = load_image(image_file)
        if original_image is None:
            continue
        original_image = original_image.to(device)
        
        # Load weights for untrained reconstruction
        model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/weights/explosion_10.ckpt")
        with torch.no_grad():
            untrained_reconstructed, _ = model(original_image)

        # Load weights for trained reconstruction
        model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/weights/no_explosion_10.ckpt")
        with torch.no_grad():
            trained_reconstructed, _ = model(original_image)

        # Display or save the images in a single grid
        show_images(original_image, untrained_reconstructed, trained_reconstructed, save_path=f"comparison_{img_number}.png")
