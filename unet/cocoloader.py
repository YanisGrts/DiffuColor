import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import color
import matplotlib.pyplot as plt

class COCOColorizationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir

        self.image_names = [f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image_np = np.array(image)

        lab_image = color.rgb2lab(image_np)

        L = lab_image[:, :, 0]
        ab = lab_image[:, :, 1:]

        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 128.0

        L_tensor = torch.tensor(L_norm, dtype=torch.float32).unsqueeze(0)
        ab_tensor = torch.tensor(ab_norm, dtype=torch.float32).permute(2, 0, 1)

        # Returns (condition, target)
        return L_tensor, ab_tensor

