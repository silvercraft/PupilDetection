import os
import torch
import shutil
import numpy as np

from PIL import Image
from model import UNet_4L
from tqdm.auto import tqdm
from torchvision.utils import save_image

import albumentations as A
from albumentations.pytorch import ToTensorV2


device = "cuda" if torch.cuda.is_available() else "cpu"



def segment(dataset_path: str, result_path: str, subjects: list, TTA: bool):

    test_transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])

    model = UNet_4L(n_channels=1, n_classes=2).to(device)
    model.load_state_dict(torch.load('./model/unet_drop_best.pt'))
    model.eval()
    
    for subject in subjects:
        print(f'{subject = }')
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            result_folder = os.path.join(result_path, subject, f'{action_number + 1:02d}')
            os.makedirs(result_folder)
            
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            for idx in tqdm(range(nr_image)):
                image_path = os.path.join(image_folder, f'{idx}.jpg')
                image = np.array(Image.open(image_path))
                image = test_transform(image=image)['image'].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    if TTA:
                        logits_1 = model(image)
                        logits_2 = model(image.flip(dims=[3])).flip(dims=[3])

                        image = image.rot90(k=1, dims=[2, 3])
                        logits_3 = model(image).rot90(k=3, dims=[2, 3])
                        logits_4 = model(image.flip(dims=[3])).flip(dims=[3]).rot90(k=3, dims=[2, 3])

                        image = image.rot90(k=1, dims=[2, 3])
                        logits_5 = model(image).rot90(k=2, dims=[2, 3])
                        logits_6 = model(image.flip(dims=[3])).flip(dims=[3]).rot90(k=2, dims=[2, 3])

                        image = image.rot90(k=1, dims=[2, 3])
                        logits_7 = model(image).rot90(k=1, dims=[2, 3])
                        logits_8 = model(image.flip(dims=[3])).flip(dims=[3]).rot90(k=1, dims=[2, 3])

                        logits = logits_1 + logits_2 + logits_3 + logits_4 + logits_5 + logits_6 + logits_7 + logits_8
                    else:
                        logits = model(image)
                    
                    pred = logits.argmax(dim=1)
                    save_image(pred.float(), os.path.join(result_folder, f'{idx}.png'))

                


if __name__ == '__main__':
    dataset_path = './dataset/public'
    result_path = './Unet_drop_noTTA'

    # subjects = ['S1', 'S2', 'S3', 'S4', 'S5']
    subjects = ['S5']

    # shutil.rmtree(result_path)
    segment(dataset_path, result_path, subjects, TTA=False)
    # segment(dataset_path, result_path, subjects, TTA=True)