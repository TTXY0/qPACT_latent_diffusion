import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models.autoencoder import AutoencoderKL


def find_min_max(file_path):
    """Find the minimum and maximum of an image dataset"""
    min = np.infty
    max = -np.infty
    with open(file_path, 'rb') as f:
        image = pickle.load(f)
    
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    image = 2 * image - 1  # Shift to [-1, 1]
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return min, max

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
    for img_number in range(N):
        print("The image number is {:d}".format(img_number))
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
    plt.yscale("log")
    # plt.ylim([-10,20])
    plt.xlabel('Index')
    plt.plot(S.cpu(), marker = 'o', markersize = 4)
    plt.savefig("svd.png")
    torch.save(S.cpu(), 'sing_vals.pt')
    torch.save(Vt.cpu(), 'Vt.pt')
    torch.save(U.cpu(), 'U.pt')
    print("The ratio of the first to the 1025th singular value is {:.4f}".format(S[1024]/S[0]))
    
    
    """
    Steps: 
    (1) Rerun this script and save Vt 
    (2) Write another script that: 
    (a) loop through the training or test sets
    (b) for each sample, encode them as you do in this script
    (c) then apply the truncated version of Vt to the mean of the encoded sample, i.e., apply V_k^T to e(x_i)
    (d) Then apply the truncated version of V, i.e., compute V_k V_k^T e(x_i)
    (f) finally, reshape back to 4 x 32 x 32 and put through the decoder 
    (g) compute error metrics on the decoded swample
    """