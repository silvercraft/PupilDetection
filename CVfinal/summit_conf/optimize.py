import os
import cv2
from matplotlib import image
import numpy as np
a=0
for i in range(1, 27):
    v2_txt = os.path.join('./v2', f'{i:02d}', 'conf.txt')
    yolo_txt = os.path.join('./yolo', f'{i:02d}', 'conf.txt')

    os.mkdir(os.path.join('./new', f'{i:02d}'))
    new_txt = os.path.join('./new', f'{i:02d}', 'conf.txt')
    new = open(new_txt, 'w')

    with open(v2_txt, 'r') as f:
        v2_conf = f.read().splitlines()
    with open(yolo_txt, 'r') as f:
        yolo_conf = f.read().splitlines()
    
    for j in range(len(v2_conf)):
        if v2_conf[j] == '0' and yolo_conf[j] == '1':
            fn = os.path.join('../Unet_drop_noTTA/S5', f'{i:02d}', f'{j}.png')
            gray = cv2.imread(fn, 0)

            output = np.zeros_like(gray)

            # step 1: calculate contours
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # step 2: Filling holes
            for contour in contours:
                cv2.drawContours(gray, [contour], 0, 255, -1)

            # step 3: Re-calculate contours
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if len(contours) == 0:
                print(0, file=new)
                continue
            
            if len(contours) == 1:
                contour = contours[0]
            
            if len(contours) > 1:
                contour_set = []
                for c in range(len(contours)):
                    contour_set.extend(contours[c].tolist())
                    contours[c]
                contour = cv2.convexHull(np.array(contour_set))
            
            if len(contour) < 5:
                print(0, file=new)
                continue

            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)
            
            # Re-load image
            gray = cv2.imread(fn, 0)
            orgin_pixel = (gray // 255).sum()
            output_pixel = (output // 255).sum()
            diff = (output_pixel + 1e-6) / (orgin_pixel + 1e-6)
            
            if 0.95 < diff < 1.05:
                # cv2.imwrite(os.path.join('./ok2', f'{i:02d}-{j:03d}.png'), output)
                a += 1
                print(1, file=new)
            else:
                print(0, file=new)
        else:
            print(v2_conf[j], file=new)
    
    new.close()

print(a)