{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original array shape: (10, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_1.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_2.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_3.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_4.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_5.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_6.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_7.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_8.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_9.npz with shape (1, 1024, 1024, 1)\n",
      "Saved /workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files/sample_10.npz with shape (1, 1024, 1024, 1)\n",
      "All files saved successfully!\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "\n",
    "# Path to the original .npz file\n",
    "original_npz_path = \"/workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/10x1024x1024x1-samples.npz\"\n",
    "\n",
    "# Directory to save the split .npz files\n",
    "output_dir = \"/workspace/thomas/latentDiffusion/autoencoderTraining/logs/Important_Logs/2025-02-24T13-05-08_latent_diffusion/checkpoints/samples/00180009/2025-03-07-11-34-11/numpy/split_files\"\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "\n",
    "# Load the original .npz file\n",
    "with np.load(original_npz_path) as data:\n",
    "    # Assuming the key for the array is 'arr_0' (default for np.savez)\n",
    "    array = data['arr_0']\n",
    "\n",
    "# Check the shape of the array\n",
    "print(f\"Original array shape: {array.shape}\")  # Should be (10, 1024, 1024, 1)\n",
    "\n",
    "# Split the array into 10 parts\n",
    "split_arrays = np.split(array, 10, axis=0)\n",
    "\n",
    "# Save each part as a separate .npz file\n",
    "for i, split_array in enumerate(split_arrays):\n",
    "    output_path = os.path.join(output_dir, f\"sample_{i+1}.npz\")\n",
    "    np.savez(output_path, split_array)\n",
    "    print(f\"Saved {output_path} with shape {split_array.shape}\")\n",
    "\n",
    "print(\"All files saved successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ml_env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
