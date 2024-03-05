import os
import cv2
import math
import shutil
import numpy as np

from tqdm import tqdm
import sys

def segment(result_path: str, new_path: str, subjects: list):
    ''' hyperparameter '''
    area_threshold = 50
    extend_threshold = 0.8
    circularity_threshold = 2
    
    sequence_idx = 0
    for subject in subjects:
        print(f'{subject = }')
        for action_number in range(len(os.listdir(os.path.join(result_path, subject)))):
            result_folder = os.path.join(result_path, subject, f'{action_number + 1:02d}')
            new_folder = os.path.join(new_path, subject, f'{action_number + 1:02d}')
            os.makedirs(new_folder, exist_ok=True)
            
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(result_folder) if name.endswith('.png')])
            for idx in range(nr_image):
                image_name = os.path.join(result_folder, f'{idx}.png')
                gray = cv2.imread(image_name, 0)

                contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                
                output = np.zeros_like(gray)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # bounding_box = cv2.boundingRect(contour)

                    # method 1: remove small noise
                    if area < area_threshold:
                        continue

                    # # bounding_box: x, y, w, h
                    # extend = area / (bounding_box[2] * bounding_box[3])

                    # # method 2: remove the contours with big extend
                    # if extend > extend_threshold:
                    #     continue
                    
                    # method 3: improving roundness
                    contour = cv2.convexHull(contour)

                    # circumference = cv2.arcLength(contour, True)
                    # circularity = circumference ** 2 / (4 * math.pi * area)

                    # # method 4: remove the circularity > 2
                    # if circularity > circularity_threshold:
                    #     continue

                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)

                orgin_pixel = (gray // 255).sum()
                output_pixel = (output // 255).sum()
                diff = (output_pixel + 1e-6) / (orgin_pixel + 1e-6)
                
                if orgin_pixel != 0 and orgin_pixel < 200:
                    # print(f'{orgin_pixel = }', image_name)
                    # cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), np.zeros_like(gray))
                    cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                    continue
                
                if diff > 1.05 or diff < 0.95:
                    print(f'{diff = }', image_name)
                    cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                    continue
                
                cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)


if __name__ == '__main__':
    result_path = sys.argv[1]
    new_path = sys.argv[2]

    # subjects = ['S1', 'S2', 'S3', 'S4']
    subjects = os.listdir(result_path)
    
    segment(result_path, new_path, subjects)