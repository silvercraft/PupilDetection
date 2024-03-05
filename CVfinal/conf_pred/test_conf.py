import os
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from torchvision import models
from conf_dataset import classify
from torch.utils.data import DataLoader


# For fast training
# torch.backends.cudnn.benchmark = True
# scaler = torch.cuda.amp.GradScaler()
device = "cuda" if torch.cuda.is_available() else "cpu"


test_tfm = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # transforms.RandomResizedCrop((1024, 1024), scale=(0.9, 1.0)),
    # transforms.RandomRotation(degrees=20),

    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

def S1_S4():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--train_image_root', type=str, default='../dataset/processed/image/all')
    parser.add_argument('--train_mask_root', type=str, default='../dataset/processed/mask/all')
    parser.add_argument('--modal_weight', type=str, default='./conf_model/best.pt')
    # hyperparameter
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')


    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(args.modal_weight))
    model = model.to(device)
    
    test_set = classify(args.train_image_root, args.train_mask_root, transform=test_tfm)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    test_accs = []

    model.eval()
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        # resnet input is 3 channels images
        images = images.repeat(1, 3, 1, 1)
        logits = model(images)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        test_accs.append(acc.item())
    
    acc = sum(test_accs) / len(test_accs)
    
    print(f"acc = {sum(test_accs)} / {len(test_accs)} = {acc:.5f}")

def S5():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, default='../dataset/public/S5')
    parser.add_argument('--modal_weight', type=str, default='./conf_model/epoch020_0.9979.pt')

    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')


    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(args.modal_weight))
    model = model.to(device)
    model.eval()

    for i in range(1, 27):
        image_folder = os.path.join(args.image_root, f'{i:02d}')
        nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])

        print(image_folder)
        f = open(f'./S5_solution/{i:02d}/conf.txt', 'w')

        for j in range(nr_image):
            filename = os.path.join(image_folder, f'{j}.jpg')
            image = test_tfm(Image.open(filename)).to(device)
            image = image.repeat(1, 3, 1, 1)
            logits = model(image)
            preds = logits.argmax(dim=-1)
            print(preds.item(), file=f)
        
        f.close()


if __name__ == '__main__':
    # S1_S4()
    S5()