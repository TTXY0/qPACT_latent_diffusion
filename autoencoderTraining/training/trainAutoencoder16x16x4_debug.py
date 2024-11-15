# This file is for Laspezia


import os
import sys
#print(sys.path)
# import torch
# import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.autoencoder16x16x4_debug import AutoencoderKL


import pickle
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import torch
# from torch.optim.lr_scheduler import LambdaLR
VER_NAME = "16x16x4"
class CustomImagePickleDataset(Dataset):
    def __init__(self, data_root, size=512):
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"The directory {data_root} does not exist.")
        self.data_root = data_root
        self.file_list = os.listdir(data_root)
        self.size = size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_root, self.file_list[idx])
        with open(file_path, 'rb') as f:
            image = pickle.load(f)

        image = image.numpy()
        
        image = (image - image.min()) / (image.max() - image.min())
        
        # Shift to [-1, 1]
        image = 2 * image - 1

        # display_image = (image + 1) / 2
        # plt.imshow(display_image, cmap='gray')
        # plt.axis('off')  
        # plt.savefig("output_image.png")
        # plt.close() 
        return {"image": image, "file_path_": file_path}
    
class ImageLoggingCallback(Callback):
    def __init__(self, every_n_steps=100):
        super().__init__()
        self.every_n_steps = every_n_steps

    def get_partials(self, tensor):
        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx, dy

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.global_step % self.every_n_steps != 0:
            return

        originals = batch["image"].to(pl_module.device).float()  # Shape: [batch_size, C, H, W]

        with torch.no_grad():
            reconstructions = []
            for i in range(originals.size(0)):  # Loop over batch
                input_img = originals[i].unsqueeze(0).unsqueeze(0).float()
                recon, _ = pl_module(input_img)
                reconstructions.append(recon.squeeze(0))
            reconstructions = torch.stack(reconstructions)

        input_partials = self.get_partials(originals.unsqueeze(0))
        recon_partials = self.get_partials(reconstructions)
        gradients_original = (torch.abs(input_partials[0]) + torch.abs(input_partials[1]))
        gradients_reconstructed = (torch.abs(recon_partials[0]) + torch.abs(recon_partials[1]))


        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

        # Original Image
        original_np = originals[0].squeeze(0).cpu().numpy() 
        axes[0, 0].imshow(original_np, vmin=-1, vmax=1)
        axes[0, 0].set_title(f"Original Image 1 vmin=-1, vmax=1")
        axes[0, 0].axis("off")

        # Reconstructed Image
        reconstructed_np = reconstructions[0].squeeze(0).cpu().numpy()
        axes[0, 1].imshow(reconstructed_np, vmin=-1, vmax=1)
        axes[0, 1].set_title(f"Reconstructed Image 1 vmin=-1, vmax=1")
        axes[0, 1].axis("off")

        difference_np = original_np - reconstructed_np
        axes[0, 2].imshow(difference_np, vmin=-1, vmax=1)
        axes[0, 2].set_title(f"Difference (Orig - Recon) vmin=-1, vmax=1")
        axes[0, 2].axis("off")

        # Gradient Original
        gradient_orig_np = gradients_original.squeeze(0).cpu()[0].squeeze(0).numpy()
        im0 = axes[1, 0].imshow(gradient_orig_np)
        axes[1, 0].set_title(f"Gradient Original 1")
        axes[1, 0].axis("off")
        fig.colorbar(im0, ax=axes[1, 0])

        # Gradient Reconstructed
        gradient_rec_np = gradients_reconstructed.squeeze(0).cpu()[0].squeeze(0).numpy()
        im1 = axes[1, 1].imshow(gradient_rec_np)
        axes[1, 1].set_title(f"Gradient Reconstructed 1")
        axes[1, 1].axis("off")
        fig.colorbar(im1, ax=axes[1, 1])

        # Original - Reconstructed)
        gradient_diff_np = gradient_orig_np - gradient_rec_np
        im2 = axes[1, 2].imshow(gradient_diff_np)
        axes[1, 2].set_title(f"Gradient Difference")
        axes[1, 2].axis("off")
        fig.colorbar(im2, ax=axes[1, 2])

        plt.tight_layout()
        trainer.logger.experiment.add_figure(f"Images/Step_{trainer.global_step}", fig, trainer.global_step)
        plt.close(fig)
        
class WeightAndGradientNormLoggingCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # weight and gradient norms for each layer
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                weight_norm = param.norm(2).item()
                trainer.logger.experiment.add_scalar(f"weight_norms/{name}", weight_norm, trainer.global_step)
                
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    trainer.logger.experiment.add_scalar(f"grad_norms/{name}", grad_norm, trainer.global_step)

if __name__ == "__main__":
    ENV = os.getenv('TRAIN_ENV', 'local')

    config_paths = {
        'super': '',
        'local': '/workspace/thomas/latentDiffusion/autoencoderTraining/configs/config_local_16x16x4.yaml',
        'bevo' : '' # Not set
    }
    
    config_path = config_paths.get(ENV, config_paths['local'])

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    seed_everything(config.get('seed', 42))
    base_lr = config['model']['base_learning_rate']
    learning_rate = base_lr

    train_dataset = CustomImagePickleDataset(data_root=config['data']['params']['train']['params']['data_root'])
    val_dataset = CustomImagePickleDataset(data_root=config['data']['params']['validation']['params']['data_root'])

    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['params']['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['data']['params']['batch_size'], shuffle=False, num_workers=4)

    if 'ckpt_path' in config['model']['params']:
        model = AutoencoderKL(**config['model']['params'])
        model.init_from_ckpt(config['model']['params']['ckpt_path'])
    else:
        model = AutoencoderKL(**config['model']['params'])
    model.learning_rate = learning_rate

    checkpoint_callback = ModelCheckpoint(monitor='val/rec_loss', save_top_k=3, mode='min')
    image_logger = ImageLoggingCallback(every_n_steps=10000)
    early_stopping_callback = EarlyStopping(monitor='val/rec_loss', patience=3, mode='min', verbose=True)
    log_dir = config['logfile']
    logger = TensorBoardLogger(log_dir, 
                           version=VER_NAME,
                           default_hp_metric=False
                           )
    weight_grad_logging = WeightAndGradientNormLoggingCallback()
    
    trainer = Trainer(
        logger=logger,
        max_epochs=config['lightning']['trainer']['max_epochs'],
        gpus=config['lightning']['trainer']['gpus'],
        precision=config['lightning']['trainer']['precision'],
        log_every_n_steps=config['lightning']['trainer']['log_every_n_steps'],
        callbacks=[checkpoint_callback, early_stopping_callback, image_logger, weight_grad_logging]
    )

    trainer.fit(model, train_dataloader, val_dataloader)
