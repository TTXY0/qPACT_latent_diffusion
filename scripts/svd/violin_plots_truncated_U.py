import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from autoencoderTraining.models.autoencoder_downsample import AutoencoderKL
from losses.ssim import ssim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics_512 = torch.load("error_metrics_k512.pt")
metrics_1024 = torch.load("error_metrics_k1024.pt")
metrics_2000 = torch.load("error_metrics_k2000.pt")
fig, ax = plt.subplots()

# Convert metrics to numpy arrays
mse_standard = np.array(metrics_512['MSE_Standard'])
ssim_standard = np.array(metrics_512['SSIM_Standard'])
mse_truncated_512 = np.array(metrics_512['MSE_Truncated'])
ssim_truncated_512 = np.array(metrics_512['SSIM_Truncated'])

mse_truncated_1024 = np.array(metrics_1024['MSE_Truncated'])
ssim_truncated_1024 = np.array(metrics_1024['SSIM_Truncated'])

mse_truncated_2000 = np.array(metrics_2000['MSE_Truncated'])
ssim_truncated_2000 = np.array(metrics_2000['SSIM_Truncated'])

# Create the figure
plt.figure(figsize=(12, 6))

# MSE Violin Plot (First Subplot)
plt.subplot(1, 2, 1)
plt.violinplot([mse_standard, mse_truncated_512,mse_truncated_1024,mse_truncated_2000], showmeans=True, showmedians=True)
plt.xticks([1, 2,3,4], ['MSE_Standard', '512', '1024','2000'])
plt.ylabel('Values')
plt.title('Distribution of MSE')


# SSIM Violin Plot (Second Subplot)
plt.subplot(1, 2, 2)
plt.violinplot([ssim_standard, ssim_truncated_512,ssim_truncated_1024,ssim_truncated_2000], showmeans=True, showmedians=True)
plt.xticks([1, 2,3,4], ['SSIM_Standard', '512', '1024','2000'])
plt.ylabel('Values')
plt.title('Distribution of SSIM')

# Save and close the plot
plt.tight_layout()
plt.savefig("error_metrics_violin_subplots.png")
plt.close()
print("Error metrics violin plot with subplots saved to 'error_metrics_violin_subplots.png'")
