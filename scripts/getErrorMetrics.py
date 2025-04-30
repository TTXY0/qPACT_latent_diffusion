import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models.autoencoder_linear_compression import AutoencoderKL as VAE_Linear_Compression
from autoencoderTraining.models.autoencoder_downsample import AutoencoderKL
from losses.ssim import ssim
import torch.nn.functional as F
import argparse

k = 512
# def get_mean(image_files):
#     all_encoded_samples = []
#     for img_number, image_file in enumerate(image_files):
#         print(f"get_mean: Processing image {img_number + 1}/{len(image_files)}")
#         image_path = os.path.join(image_dir, image_file)
#         original_image = load_image(image_path).to(device)
#         with torch.no_grad():
#             encoded = model.encode(original_image).mean.flatten()  # (4096,)
#             all_encoded_samples.append(encoded)
#     all_encoded_samples = torch.stack(all_encoded_samples)  # (N, 4096)
#     encoded_mean = all_encoded_samples.mean(dim=0)  # (4096,)
#     return encoded_mean

def load_image(file_path):
    """Load and preprocess an image from a pickle file."""
    with open(file_path, 'rb') as f:
        image = pickle.load(f)
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    # Normalize to [-1, 1]
    image = 2 * ((image - image.min()) / (image.max() - image.min())) - 1
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def show_images(original, untrained_reconstructed, trained_reconstructed, save_path=None):
    original_np = original.squeeze(0).squeeze(0).cpu().numpy()
    untrained_np = untrained_reconstructed.squeeze(0).squeeze(0).cpu().numpy()
    trained_np = trained_reconstructed.squeeze(0).squeeze(0).cpu().numpy()
    difference_np = trained_np - original_np


    images = [original_np, untrained_np, trained_np, difference_np]
    titles = ["Original Image", "Reconstructed (Standard)", "Reconstructed (Truncated U)", "Difference (Truncated U - Original)"]
    vmin, vmax = -1, 1 

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    combined_image = np.concatenate(
        [((img - vmin) / (vmax - vmin)) for img in images], axis=1
    ) 

    im = ax.imshow(combined_image, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.axis("off")

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
    config_path1 = '/workspace/thomas/latentDiffusion/autoencoderTraining/configs/config_local_k512.yaml'
    with open(config_path1, 'r') as file:
        config1 = yaml.safe_load(file)
        
    config_path2 = '/workspace/thomas/latentDiffusion/autoencoderTraining/configs/config_local_f8.yaml'
    with open(config_path2, 'r') as file:
        config2 = yaml.safe_load(file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE_Linear_Compression(**config1['model']['params']).to(device)
    model.eval()
    model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/weights/linearCompression/U512_5.ckpt")
    
    untrained_model = AutoencoderKL(**config2['model']['params']).to(device)
    untrained_model.eval()
    untrained_model.init_from_ckpt("/workspace/thomas/latentDiffusion/autoencoderTraining/weights/no_explosion_10.ckpt")

    U_path = 'U.pt'
    S_path = 'sing_vals.pt'
    U = torch.load(U_path).to(device)  # (4096, 2000)
    # S = torch.load(S_path).to(device)  # (4096,)

    U_k = U[:, :k]  # (4096, 2000)

    image_dir = '/workspace/thomas/latentDiffusion/autoencoderTraining/data/MuaSlices/train'
    image_files = os.listdir(image_dir)[:10000]  # 10000 samples
    #encoded_mean = get_mean(image_files)

    metrics = {'MSE_Standard': [], 'MSE_Truncated': [], 'SSIM_Standard': [], 'SSIM_Truncated': []}

    for img_number, image_file in enumerate(image_files):
        print(f"Processing image {img_number + 1}/{len(image_files)}")
        image_path = os.path.join(image_dir, image_file)
        original_image = load_image(image_path).to(device)
        if original_image is None:
            print(f"Warning: Failed to load image {image_file}. Skipping.")
            continue
        # TODO: make encoded_mean the mean of N encoded samples
        with torch.no_grad():
            # Standard reconstruction *
            reconstructed_standard = untrained_model(original_image)[0]  # (4096,)
            #decoded_standard = model.decode(encoded_standard)  

            reconstructed = model(original_image)[0]

        # print(reconstructed_standard[0], "\n\n")
        # print(reconstructed_standard[1])
        mse_standard = F.mse_loss(reconstructed_standard, original_image).item()
        ssim_standard = ssim(reconstructed_standard, original_image, data_range=2.0).item()
        metrics['MSE_Standard'].append(mse_standard)
        metrics['SSIM_Standard'].append(ssim_standard)


        mse_truncated = F.mse_loss(reconstructed, original_image).item()
        ssim_truncated = ssim(reconstructed, original_image, data_range=2.0).item()
        metrics['MSE_Truncated'].append(mse_truncated)
        metrics['SSIM_Truncated'].append(ssim_truncated)

        # Save Images
        if img_number < 5:  
            save_path = f"U_k={k}_{img_number+1}.png"
            show_images(original_image, reconstructed_standard, reconstructed, save_path=save_path)
            print(f"Saved comparison image to {save_path}")


    torch.save(metrics, f'error_metrics_k{k}_trained.pt')
    print("Error metrics saved to 'error_metrics.pt'")

    plt.figure(figsize=(12, 5))

    # MSE Plot
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(metrics['MSE_Standard'])), metrics['MSE_Standard'], label='MSE Standard', alpha=0.5)
    plt.scatter(range(len(metrics['MSE_Truncated'])), metrics['MSE_Truncated'], label='MSE Truncated', alpha=0.5)
    plt.xlabel('Image Index')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error')
    plt.legend()

    # SSIM Plot
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(metrics['SSIM_Standard'])), metrics['SSIM_Standard'], label='SSIM Standard', alpha=0.5, color='orange')
    plt.scatter(range(len(metrics['SSIM_Truncated'])), metrics['SSIM_Truncated'], label='SSIM Truncated', alpha=0.5, color='green')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM')
    plt.title('Structural Similarity Index')
    plt.legend()

    plt.tight_layout()
    plt.savefig("error_metrics_scatter.png")
    plt.close()
    print("Error metrics scatter plot saved to 'error_metrics_scatter.png'")
