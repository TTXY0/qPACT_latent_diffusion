from torch.utils.data import DataLoader, Dataset
import os
import pickle
import numpy as np

class CustomImagePickleDataset(Dataset):
    def __init__(self, data_root):
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"The directory {data_root} does not exist.")
        self.data_root = data_root
        self.file_list = os.listdir(data_root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_root, self.file_list[idx])
        with open(file_path, 'rb') as f:
            image = pickle.load(f)

        image = image.numpy()
        return {"image": image, "file_path_": file_path}
    

def save_npz_images(data_root, output_dir, num_images=5):
    dataset = CustomImagePickleDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):
        if i >= num_images:
            break
        
        image = batch["image"].numpy()  # Extract image tensor
        file_name = os.path.basename(batch["file_path_"][0])  # Get original filename
        #np_image = custom_to_np(image)  # Convert to numpy format

        save_path = os.path.join(output_dir, f"{file_name}.npz")
        np.savez_compressed(save_path, image=image)  # Save as npz

        print(f"Saved {save_path}")

# Example usage
data_root = "/workspace/thomas/latentDiffusion/autoencoderTraining/data/MuaSlices/train"
output_dir = "/workspace/thomas/latentDiffusion/autoencoderTraining/data/npz_samples"
save_npz_images(data_root, output_dir)