import os
import cv2
from cv2 import ellipse
import numpy as np

result_path = './trash/pupil_detection/yolov5/runs/detect'
gray = cv2.imread('./dataset/public/S5/01/0.png', 0)
H, W = gray.shape

for i in range(1, 27):
    if i == 1:
        folder = os.path.join(result_path, 'exp')
    else:
        folder = os.path.join(result_path, f'exp{i}')
    nr_image = len([name for name in os.listdir(folder) if name.endswith('.jpg')])

    os.makedirs(f'./yolo_seg/S5/{i:02d}', exist_ok=True)
    os.makedirs(f'./yolo_conf/S5/{i:02d}', exist_ok=True)
    
    
    f = open(f'./yolo_conf/S5/{i:02d}/conf.txt', 'w')
    for j in range(nr_image):
        output = np.zeros_like(gray)
        if i == 1:
            bbox_txt = os.path.join(result_path, f'exp', 'labels', f'{j}.txt')
        else:
            bbox_txt = os.path.join(result_path, f'exp{i}', 'labels', f'{j}.txt')
        # print(bbox_txt)
        if os.path.exists(bbox_txt):
            print(1, file=f)
            with open(bbox_txt, 'r') as cf:
                data = cf.read().split()
            x_center, y_center, width, height = float(data[1]), float(data[2]), float(data[3]), float(data[4])
            # print(x_center, y_center, width, height)
            points = [
                [[round(x_center * W), round(y_center * H)]],
                [[round(x_center * W), round((y_center - height / 2) * H)]],
                [[round((x_center + width / 2) * W), round(y_center * H)]],
                [[round(x_center * W), round((y_center + height / 2) * H)]],
                [[round((x_center - width / 2) * W), round(y_center * H)]],
            ]

            ellipse = cv2.fitEllipse(np.array(points))
            cv2.ellipse(output, box=ellipse, color=255, thickness=-1)
        else:
            print(0, file=f)
        cv2.imwrite(f'./yolo_seg/S5/{i:02d}/{j}.png', output)

    f.close()