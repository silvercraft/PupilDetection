import os
import wandb
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision import models
from conf_dataset import classify
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


# For fast training
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
device = "cuda" if torch.cuda.is_available() else "cpu"


train_tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop((1024, 1024), scale=(0.9, 1.0)),
    transforms.RandomRotation(degrees=20),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

valid_tfm = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--train_image_root', type=str, default='../dataset/processed/image/train')
    parser.add_argument('--train_mask_root', type=str, default='../dataset/processed/mask/train')
    parser.add_argument('--valid_image_root', type=str, default='../dataset/processed/image/valid')
    parser.add_argument('--valid_mask_root', type=str, default='../dataset/processed/mask/valid')
    parser.add_argument('--modal_path', type=str, default='./conf_model')
    # hyperparameter
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # wandb
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    
    if args.wandb:
        wandb.init(project="CVfinal-conf", name='resnet34_v2', entity='freeway', config=args)


    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2)
    model = model.to(device)
    
    train_set = classify(args.train_image_root, args.train_mask_root, transform=train_tfm)
    valid_set = classify(args.valid_image_root, args.valid_mask_root, transform=valid_tfm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / args.warmup_epoch)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, min_lr=1e-6)
    
    for epoch in range(1, args.epochs + 1):
        print(f'{epoch = }')

        if args.wandb:
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)
    
        train_loss = []
        train_acc = 0.0

        model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # resnet input is 3 channels images
            images = images.repeat(1, 3, 1, 1)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())
            acc = (logits.argmax(dim=-1) == labels).float().sum()
            train_acc += acc
    
        train_loss = sum(train_loss) / len(train_loss)
        train_acc /= len(train_set)
        
        print(f"train loss = {train_loss:.5f}, train acc = {train_acc:.5f}")

        valid_loss = []
        valid_acc = 0.0

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(device), labels.to(device)
                
                # resnet input is 3 channels images
                images = images.repeat(1, 3, 1, 1)

                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)

                valid_loss.append(loss.item())
                acc = (logits.argmax(dim=-1) == labels).float().sum()
                valid_acc += acc
    
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc /= len(valid_set)
        
        print(f"valid loss = {valid_loss:.5f}, valid acc = {valid_acc:.5f}")

        torch.save(model.state_dict(), os.path.join(args.modal_path, f'v2_epoch{epoch:03d}_{valid_acc:.4f}.pt'))
        
        if args.wandb:
            wandb.log({
                'train loss': train_loss,
                'train acc': train_acc,
                'valid loss': valid_loss,
                'valid acc': valid_acc,
            }, step=epoch)
        
        if epoch < args.warmup_epoch:
            warmup.step()
        else:
            scheduler.step(valid_acc)


if __name__ == '__main__':
    main()