import os
# import wandb
import torch
import argparse
import torch.nn as nn

from metric import IoU
from model import UNet_4L, AutoEncoder_4L
from tqdm.auto import tqdm
from mydataset import myDataset_2
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


# For fast training
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
# path
parser.add_argument('--train_image_root', type=str, default='./dataset/processed/image/all')
parser.add_argument('--train_mask_root', type=str, default='./dataset/processed/mask/all')
# parser.add_argument('--train_image_root', type=str, default='./dataset/processed/image/train')
# parser.add_argument('--valid_image_root', type=str, default='./dataset/processed/image/valid')
# parser.add_argument('--train_mask_root', type=str, default='./dataset/processed/mask/train')
# parser.add_argument('--valid_mask_root', type=str, default='./dataset/processed/mask/valid')
parser.add_argument('--modal_path', type=str, default='./model')
parser.add_argument('--autoencoder_weights', type=str, default='./model/AE_best.pt')
# hyperparameter
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100, help="Number of total epochs")
parser.add_argument('--warmup_epoch', type=int, default=10, help="Number of warmup epochs")
parser.add_argument('--batch_size', type=int, default=8, help="Mini-batch size")
parser.add_argument('--num_workers', type=int, default=0, help="How many subprocesses to use for data loading")
# # wandb
# parser.add_argument('--wandb', action='store_true')

args = parser.parse_args()
for arg in vars(args):
    print(f'{arg} = {getattr(args, arg)}')

# if args.wandb:
#     wandb.init(project="CVfinal", name='Unet4L_AE_drop', entity='freeway', config=args)

print('\nPreparing Dataset and DataLoader...')
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2(),
])

# valid_transform = A.Compose([
#     A.Normalize(mean=0.5, std=0.5),
#     ToTensorV2(),
# ])

train_dataset = myDataset_2(args.train_image_root, args.train_mask_root, train_transform)
# valid_dataset = myDataset(args.valid_image_root, args.valid_mask_root, valid_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
# valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

# if args.wandb:
#     wandb.config.train_num = len(train_dataset)
#     # wandb.config.valid_num = len(valid_dataset)

print('Creating model...')
model = UNet_4L(n_channels=1, n_classes=2)
model_dict = model.state_dict()

autoencoder = AutoEncoder_4L(n_channels=1)
autoencoder.load_state_dict(torch.load(args.autoencoder_weights))
autoencoder_dict = autoencoder.state_dict()

for i, autoencoder_key in enumerate(autoencoder_dict.keys()):
    if i == 60:
        break
    if i >= 12:
        new_key = autoencoder_key.replace('encode', 'down')
    else:
        new_key = autoencoder_key
    model_dict[new_key] = autoencoder_dict[autoencoder_key]

model.load_state_dict(model_dict)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
warmup = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / args.warmup_epoch)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, min_lr=1e-6)

print('Start training...')
best_iou = 0.0
for epoch in range(1, args.epochs + 1):
    print(f'{epoch = }')

    # if args.wandb:
    #     wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

    train_loss = 0
    # valid_loss = 0
    
    model.train()
    train_iou = []
    for image, mask in tqdm(train_loader):
        image, mask = image.to(device), mask.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(image)
            loss = criterion(logits, mask)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()

        pred = logits.argmax(dim=1)
        train_iou.extend(IoU(pred, mask))
    
    # model.eval()
    # valid_iou = []
    # with torch.no_grad():
    #     for image, mask in tqdm(valid_loader):
    #         image, mask = image.to(device), mask.to(device)
            
    #         with torch.cuda.amp.autocast():
    #             logits = model(image)
    #             loss = criterion(logits, mask)
            
    #         valid_loss += loss.item()

    #         pred = logits.argmax(dim=1)
    #         valid_iou.extend(IoU(pred, mask))
    
    train_loss /= len(train_loader)
    train_iou = sum(train_iou) / len(train_iou)
    # valid_loss /= len(valid_loader)
    # valid_iou = sum(valid_iou) / len(valid_iou)



    # print(f'{train_loss = }   {valid_loss = }   {valid_iou = }')

    # if valid_iou > best_iou:
    #     best_iou = valid_iou
    #     torch.save(model.state_dict(), os.path.join(args.modal_path, f'best_model.pt'))
    # else:
    #     torch.save(model.state_dict(), os.path.join(args.modal_path, f'last_model.pt'))

    if train_iou > best_iou:
        best_iou = train_iou
        torch.save(model.state_dict(), os.path.join(args.modal_path, f'unet_drop_best.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(args.modal_path, f'unet_drop_last.pt'))

    # if args.wandb:
    #     wandb.log({
    #         'train loss': train_loss,
    #         'train IOU': train_iou,
    #         # 'valid loss': valid_loss,
    #         # 'valid IOU': valid_iou,
    #     }, step=epoch)
    
    if epoch < args.warmup_epoch:
        warmup.step()
    else:
        # scheduler.step(valid_iou)
        scheduler.step(train_iou)

# if args.wandb:
#     wandb.finish()