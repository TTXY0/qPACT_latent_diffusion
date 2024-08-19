import os
import torch
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from ldm.util import instantiate_from_config
from ldm.modules.losses import LPIPSWithDiscriminator
import pickle
import numpy as np


class Autoencoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = instantiate_from_config(config['model'])
        self.learning_rate = config['model']['base_learning_rate']
        self.criterion = LPIPSWithDiscriminator()  # Replace with your loss function if different

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        outputs = self(images)
        loss = self.criterion(outputs, images)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        outputs = self(images)
        loss = self.criterion(outputs, images)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CustomImagePickleDataset(Dataset):
    def __init__(self, data_root, size=512):
        self.data_root = data_root
        self.file_list = os.listdir(data_root)
        self.size = size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(os.path.join(self.data_root, self.file_list[idx]), 'rb') as f:
            image = pickle.load(f)

        # Convert to NumPy array
        image = image.numpy()
        
        # Normalize to uint8 (0-255)
        image = (image * 255).astype(np.uint8)
        
        # Convert grayscale to RGB
        image = np.stack([image] * 3, axis=-1)

        # Resize and convert to tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW

        return {"image": image, "file_path_": os.path.join(self.data_root, self.file_list[idx])}


if __name__ == "__main__":
    config_path = 'config.yaml'  # Replace with your YAML config path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Seed everything for reproducibility
    seed_everything(config.get('seed', 42))

    # Instantiate datasets
    train_dataset = CustomImagePickleDataset(data_root=config['data']['train_data_root'])
    val_dataset = CustomImagePickleDataset(data_root=config['data']['val_data_root'])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=4)

    # Initialize model
    model = Autoencoder(config)

    # Callbacks for checkpointing and learning rate monitoring
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = Trainer(
        max_epochs=config.get('max_epochs', 10),
        gpus=config.get('gpus', 1),  # Adjust based on your setup
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16 if config.get('use_amp', False) else 32,
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
