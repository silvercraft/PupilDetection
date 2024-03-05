import os
import cv2
import shutil


dataset_path = "./dataset/public"
subjects = ['S1', 'S2', 'S3', 'S4']

os.makedirs('./dataset/processed/image/all', exist_ok=True)
os.makedirs('./dataset/processed/image/train', exist_ok=True)
os.makedirs('./dataset/processed/image/valid', exist_ok=True)
os.makedirs('./dataset/processed/mask/all', exist_ok=True)
os.makedirs('./dataset/processed/mask/train', exist_ok=True)
os.makedirs('./dataset/processed/mask/valid', exist_ok=True)

images = []
masks = []
for subject in subjects:
    for action_number in range(26):
        image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
        nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
        for idx in range(nr_image):
            images.append([os.path.join(image_folder, f'{idx}.jpg'), f'{subject}_{action_number + 1:02d}_{idx:03d}.jpg'])
            masks.append([os.path.join(image_folder, f'{idx}.png'), f'{subject}_{action_number + 1:02d}_{idx:03d}.png'])

for i in range(len(images)):
    phase = 'valid' if i % 10 == 0 else 'train'
    shutil.copyfile(images[i][0], os.path.join(f'./dataset/processed/image/{phase}', images[i][1]))
    shutil.copyfile(images[i][0], os.path.join(f'./dataset/processed/image/all', images[i][1]))

    gray = cv2.imread(masks[i][0], 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(f'./dataset/processed/mask/{phase}', masks[i][1]), th)
    cv2.imwrite(os.path.join(f'./dataset/processed/mask/all', masks[i][1]), th)