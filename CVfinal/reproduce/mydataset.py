import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class myDataset_2(Dataset):
    def __init__(self, image_root, mask_root, transform):
        self.images = []
        self.masks = []
        self.transform = transform

        image_paths = sorted([os.path.join(image_root, fn) for fn in os.listdir(image_root)])
        mask_paths = sorted([os.path.join(mask_root, fn) for fn in os.listdir(mask_root)])
        
        for image_path, mask_path in zip(image_paths, mask_paths):
            mask = np.array(Image.open(mask_path)) / 255
            
            if not mask.any():
                continue
            
            image = np.array(Image.open(image_path))
            self.images.append(image)
            self.masks.append(mask)

    def __getitem__(self, idx):
        transformed = self.transform(image=self.images[idx], mask=self.masks[idx])
        image = transformed['image']
        mask = transformed['mask'].long()

        return image, mask
    
    def __len__(self):
        return len(self.images)

class myDataset(Dataset):
    def __init__(self, image_root, mask_root, transform):
        self.images = []
        self.masks = []
        self.transform = transform

        image_paths = sorted([os.path.join(image_root, fn) for fn in os.listdir(image_root)])
        mask_paths = sorted([os.path.join(mask_root, fn) for fn in os.listdir(mask_root)])

        for image_path in image_paths:
            image = np.array(Image.open(image_path))
            self.images.append(image)
        
        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path)) / 255
            self.masks.append(mask)

    def __getitem__(self, idx):
        transformed = self.transform(image=self.images[idx], mask=self.masks[idx])
        image = transformed['image']
        mask = transformed['mask'].long()

        return image, mask
    
    def __len__(self):
        return len(self.images)

class AutoEncoder_Dataset(Dataset):
    def __init__(self, public_root, transform):
        self.images = []
        self.transform = transform
        
        subjects = ['S1', 'S2', 'S3', 'S4', 'S5']
        for subject in subjects:
            for action_number in range(26):
                image_folder = os.path.join(public_root, subject, f'{action_number + 1:02d}')
                nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
                for idx in range(nr_image):
                    self.images.append(np.array(Image.open(os.path.join(image_folder, f'{idx}.jpg'))))


    def __getitem__(self, idx):
        return self.transform(image=self.images[idx])['image']
    
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightnessContrast(),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])
    train_dataset = myDataset_2('./dataset/processed/image/all', './dataset/processed/mask/all', train_transform)