import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm


dataset_path = '../CVfinal/dataset/public'
subjects = ['S1', 'S2', 'S3', 'S4']

ID = 1
for subject in subjects:
    print(f'{subject = }')
    for action_number in range(26):
        image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
        nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
        for idx in tqdm(range(nr_image)):
            # images
            src = os.path.join(image_folder, f'{idx}.jpg')
            dst = os.path.join('./datasets/pupil/images', f'{ID:05d}.jpg')
            shutil.copyfile(src, dst)


            # ground truth
            src = os.path.join(image_folder, f'{idx}.png')
            dst = os.path.join('./datasets/pupil/labels', f'{ID:05d}.txt')

            image = cv2.imread(src)

            is_all_zero = np.all(image == 0)
            if is_all_zero:
                ID += 1
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            # contours only has one element
            x, y, w, h = cv2.boundingRect(contours[0])
            
            x_center = (x + (w - 1) / 2) / 639
            y_center = (y + (h - 1) / 2) / 479
            width = w / 640
            height = h / 480

            with open(dst, 'w') as f:
                f.write(f'0 {x_center} {y_center} {width} {height}')
            
            ID += 1