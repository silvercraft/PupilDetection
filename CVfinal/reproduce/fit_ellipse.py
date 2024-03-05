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
                for contour in contours:
                    cv2.drawContours(gray, [contour], 0, 255, -1)
                contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                
                output = np.zeros_like(gray)

                if len(contours) == 0:
                    cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                    continue
                
                if len(contours) == 1:
                    contour = cv2.convexHull(contours[0])

                    if len(contour) < 5:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue

                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(output, box=ellipse, color=255, thickness=-1)

                    orgin_pixel = (gray // 255).sum()
                    output_pixel = (output // 255).sum()
                    diff = (output_pixel + 1e-6) / (orgin_pixel + 1e-6)
                    
                    if orgin_pixel != 0 and orgin_pixel < area_threshold:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue
                    
                    if output_pixel > 8000:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue
                    
                    if diff > 1.05 or diff < 0.95:
                        print(f'{diff = }', image_name)
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                        continue

                    cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                    continue
                
                contour_set = []
                for i in range(len(contours)):
                    contour_set.extend(contours[i].tolist())
                contour = cv2.convexHull(np.array(contour_set))

                if len(contour) < 5:
                    cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                    continue

                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(output, box=ellipse, color=255, thickness=-1)

                orgin_pixel = (gray // 255).sum()
                output_pixel = (output // 255).sum()
                diff = (output_pixel + 1e-6) / (orgin_pixel + 1e-6)
                
                if orgin_pixel != 0 and orgin_pixel < area_threshold:
                    cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                    continue
                    
                if output_pixel > 8000:
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
    subjects = [sys.argv[3]]
    
    segment(result_path, new_path, subjects)