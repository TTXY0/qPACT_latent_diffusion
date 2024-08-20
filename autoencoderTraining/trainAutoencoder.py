import os
import sys
print(sys.path)
import torch
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from ldm.models.autoencoder import VQModel
from ldm.modules.losses import LPIPSWithDiscriminator
import pickle
import numpy as np

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

        # Assuming the pickle file contains a NumPy array
        image = image.numpy()
        
        # Normalize to uint8 (0-255)
        image = (image * 255).astype(np.uint8)
        
        # Convert grayscale to RGB
        image = np.stack([image] * 3, axis=-1)

        # Convert to tensor and reshape to CHW
        # image = torch.tensor(image, dtype=torch.float32)#.permute(2, 0, 1)

        return {"image": image, "file_path_": file_path}


if __name__ == "__main__":
    config_path = 'config.yaml'  # Replace with your YAML config path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Seed everything for reproducibility
    seed_everything(config.get('seed', 42))
    
    # Calculate learning rate
    bs = config['data']['params']['batch_size']
    base_lr = config['model']['base_learning_rate']
    ngpu = 1  
    accumulate_grad_batches = config.get('lightning', {}).get('trainer', {}).get('accumulate_grad_batches', 1)
    
    if config.get('opt', {}).get('scale_lr', True):
        learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    else:
        learning_rate = base_lr

    # Instantiate datasets
    train_dataset = CustomImagePickleDataset(data_root=config['data']['params']['train']['params']['data_root'])
    val_dataset = CustomImagePickleDataset(data_root=config['data']['params']['validation']['params']['data_root'])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['data']['params']['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config['data']['params']['batch_size'], shuffle=False, num_workers=4)

    # Initialize model
    model = VQModel(**config['model']['params'])
    model.learning_rate = learning_rate

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
