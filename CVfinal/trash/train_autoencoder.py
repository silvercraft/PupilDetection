import os
# import wandb
import torch
import argparse
import torch.nn as nn

from tqdm.auto import tqdm
from model import AutoEncoder_4L
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from mydataset import AutoEncoder_Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# For fast training
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'{device = }')


parser = argparse.ArgumentParser()
# path
parser.add_argument('--public_dir', type=str, default='./dataset/public')
parser.add_argument('--model_dir', type=str, default='./model')
# hyperparameter
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--epochs', type=int, default=100, help="Number of total epochs")
parser.add_argument('--warmup_epoch', type=int, default=10, help="Number of warmup epochs")
parser.add_argument('--batch_size', type=int, default=8, help="Mini-batch size")
parser.add_argument('--num_workers', type=int, default=0, help="How many subprocesses to use for data loading")
# wandb
parser.add_argument('--wandb', action='store_true')

args = parser.parse_args()
for arg in vars(args):
    print(f'{arg} = {getattr(args, arg)}')

# if args.wandb:
#     wandb.init(project='pupil SSL', name='AE_4L', entity='freeway', config=args)

print('Preparing Dataset and DataLoader...')
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2(),
])

train_dataset = AutoEncoder_Dataset(public_root=args.public_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

# if args.wandb:
#     wandb.config.train_num = len(train_dataset)

print('Creating model...')
model = AutoEncoder_4L(n_channels=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
warmup = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / args.warmup_epoch)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=10, min_lr=1e-6)

print('Start training...')
best_loss = 100
for epoch in range(1, args.epochs + 1):
    print(f'{epoch = }')

    # if args.wandb:
    #     wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

    train_loss = 0
    
    model.train()
    for image in tqdm(train_loader):
        image = image.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            reconstruct = model(image)
            loss = criterion(reconstruct, image)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), os.path.join(args.model_dir, f'AE_best.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(args.model_dir, f'AE_last.pt'))
    
    print(f'loss = {train_loss:.4f}')

    # if args.wandb:
    #     wandb.log({'train loss': train_loss}, step=epoch)
    
    if epoch < args.warmup_epoch:
        warmup.step()
    else:
        scheduler.step(train_loss)

# if args.wandb:
#     wandb.finish()