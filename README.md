# Latent Diffusion Modeling for Photoacoustic Computed Tomography (PACT)

Latent Diffusion Model (LDM) using Singular Value Decomposition based dimensionality reduction.

---

## Overview

Please see project_summary.pdf

By encoding log(µₐ) maps into a structured latent space, applying singular value decomposition (SVD) to reduce dimensionality, and training a diffusion model in this compressed space, we enable efficient sampling, denoising, and CRB estimation. This repository contatins the training, sampling, and analysis pipelines for the LDM.

---

## Repository Structure

```text
root/                  
├── configs/            # configurations for autoencoder and diffusion models             
├── jobs/               # batch scripts for training on TACC (HPC)   
├── losses/             # sample from the trained diffusion model      
├── models/             # autoencoder and diffusion PyTorch models
├── scripts/            # scripts for analysis, SVD, and sampling
├── training/           # training scripts 
├── stable-diffusion/   # stable diffusion submodule                                     
└── README.md           # this file    
```
                              

---

## Dataset

- **Source**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OZRVX6  
- See project_summary.pdf for dataset generation

---

## Usage

### 1. Install Dependencies

See the README.md in the stable diffusion repository for environemnt setup

---

### 2. Train the Autoencoder

cd autoencoderTraining  
python trainAutoencoder.py

- Adapts the Stable Diffusion autoencoder with custom layers.  
- Uses a loss function combining MS-SSIM, ℓ1 loss, and KL divergence.  

---

### 4. Train the Latent Diffusion Model

cd diffusionTraining  
python main.py \
  -t \
  --base [config path] \
  --scale_lr \
  --name latent_diffusion

- Projects reduced latent back to full latent space for diffusion.  
- Learns the score function ∇z log q(z) via a U-Net.  

---

### 5. Sample from the Trained Diffusion Model

cd sampling  
python scripts/sample_diffusion.py \
  --resume [ckpt path]
  --n_samples 1000 \
  --eta 1.0 \   
  --custom_steps 100 \ 
  --vanilla_sample True \
  --logdir ./stochastic_samples

- Generates denoised latent samples and decodes them into images.  
- Outputs images to outputs/ or specified directory.  

---


## Acknowledgments

This project builds on:  
- Stable Diffusion – CompVis  
- Harvard Breast Phantoms Dataset (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OZRVX6)  

Portions of this code are adapted from the above sources under their respective licenses.

This readme file is still being drafted
