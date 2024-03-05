import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import os
from torchvision.datasets import DatasetFolder
import cv2



class classify(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.transform = transform

        # read filenames
        label_dir = root.replace("img", "white_mask")
        labels = sorted(os.listdir(label_dir))
        for idx, img_name in enumerate(sorted(os.listdir(root))):
            filename = os.path.join(root, img_name)

            label_map = cv2.imread(os.path.join(label_dir, labels[idx]))
            # print(len(np.unique(label_map)), np.unique(label_map))
            if len(np.unique(label_map))==1:
                label=0
            else:
                label=1
            # label = int(img_name.split('_')[0])
            self.filenames.append((filename, label)) 
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
        
class classify_test(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.transform = transform

        # read filenames
        for idx, dir_name in enumerate(sorted(os.listdir(root))):
            imgs = os.listdir(os.path.join(root, dir_name))
            print(root.split('/')[-1])
            if root.split('/')[-1]=='S5':
                num = len(imgs)
            else:
                num = len(imgs)//2
            for img_name in range(num):
                filename = os.path.join(root, dir_name, str(img_name)+'.jpg')
                self.filenames.append(filename) 
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)

        # dir_fn = image_fn.split('/')[-2]
        return image, image_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class classify_test_ch(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.transform = transform

        # read filenames
        for idx, img_name in enumerate(sorted(os.listdir(root))):
            img_path = os.path.join(root, img_name)
            
            self.filenames.append(img_path) 
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)

        # dir_fn = image_fn.split('/')[-2]
        return image, image_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len