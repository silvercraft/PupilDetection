import os
import cv2

# path = './Unet_noTTA_test_fit/S5_solution'
path = './B/submit_5'

for action_number in range(26):
    folder = os.path.join(path, f'{action_number + 1:02d}')
    nr_image = len([name for name in os.listdir(folder) if name.endswith('.png')])

    f = open(os.path.join(folder, 'conf.txt'), 'w')

    for idx in range(nr_image):
        image_name = os.path.join(folder, f'{idx}.png')
        gray = cv2.imread(image_name, 0)
        if gray.sum() != 0:
            print(1, file=f)
        else:
            print(0, file=f)
    
    f.close()