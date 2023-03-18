import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ElectrophoresisDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        mask_path = os.path.join(self.mask_dir, self.image_list[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image, mask = self.transform(image, mask)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        mask = mask.astype(np.int64)
        mask = torch.from_numpy(mask)

        return image, mask

# Create an instance of the ElectrophoresisDataset class
image_dir = "path/to/your/images"
mask_dir = "path/to/your/masks"
dataset = ElectrophoresisDataset(image_dir, mask_dir)

# Create a DataLoader for efficient batching
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader to access image and mask batches
for batch_idx, (images, masks) in enumerate(dataloader):
    # `images` and `masks` are now tensors with shape (batch_size, 1, height, width)
    print("Batch index:", batch_idx)
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)
