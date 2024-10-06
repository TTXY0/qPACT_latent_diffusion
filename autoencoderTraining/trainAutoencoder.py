import os
import sys
#print(sys.path)
# import torch
# import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from autoencoder import AutoencoderKL
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import LambdaLR

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



if __name__ == "__main__":
    config_path = '/workspace/thomas/latentDiffusion/autoencoderTraining/config.yaml' 
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
    early_stopping_callback = EarlyStopping(monitor='val/rec_loss', patience=3, mode='min', verbose=True)

    trainer = Trainer(
        max_epochs=config['lightning']['trainer']['max_epochs'],
        gpus=config['lightning']['trainer']['gpus'],
        precision=config['lightning']['trainer']['precision'],
        log_every_n_steps=config['lightning']['trainer']['log_every_n_steps'],
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, train_dataloader, val_dataloader)
