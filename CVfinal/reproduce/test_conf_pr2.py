import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
from torch.utils.data import Dataset, DataLoader
import glob
import os
from torchvision.datasets import DatasetFolder
from torchvision import models
from pr_conf_dataset import classify_test
from pr_conf_dataset import classify
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

test_tfm = transforms.Compose([
    transforms.Resize((1980, 1980)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    same_seeds(30)
    model = models.resnet34(pretrained=True)
    # model.fc = nn.Sequential(nn.Linear(512, 128),
    #                         nn.ReLU(),
    #                         nn.Linear(128, 2))
    model.fc = nn.Linear(512, 2)
    if torch.cuda.is_available():
      model = model.cuda()
    checkpoint = torch.load(config.model_path)
    model = checkpoint['model']
    
    batch_size = 1
    test_set = classify_test(root=config.img_dir, transform=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,  pin_memory=True)
    
    model.eval()
    predictions = {}
    accs = []
    img_name = []
    image_paths = []

    for batch in test_loader:
        imgs, img_paths = batch
        
        with torch.no_grad():
            logits = model(imgs.to(device))
        
        #acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        #accs.append(acc)
        img_paths = img_paths[0]
        print(img_paths)
        
        key = '/'.join(img_paths.split('/')[-3:-1])
        print(key)
        if key not in predictions:
            predictions[key] = logits.argmax(dim=-1).cpu().numpy().tolist()
        else:
            predictions[key].extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        
    with open(config.save_path, 'w') as f:
        json.dump(predictions, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='./data/public/S5')
    parser.add_argument('--save_path', type=str, default='./conf_pred/0613_s5.json')
    parser.add_argument('--model_path', default='./conf_model/0613.pth', type=str, help='model path.')
    
    config = parser.parse_args()
    # print(config)
    main(config)