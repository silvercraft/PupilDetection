import os
import cv2

from PIL import Image
from torch.utils.data import Dataset


class classify(Dataset):
    def __init__(self, image_root, mask_root, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for img_name in sorted(os.listdir(image_root)):
            filename = os.path.join(image_root, img_name)
            self.images.append(Image.open(filename))
        
        for img_name in sorted(os.listdir(mask_root)):
            filename = os.path.join(mask_root, img_name)
            mask = cv2.imread(filename)
            self.labels.append(1 if mask.any() else 0)
                              
    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
            
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    image_root = '../dataset/processed/image/all'
    mask_root = '../dataset/processed/mask/all'
    
    train_dataset = classify(image_root, mask_root)