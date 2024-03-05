import os
import cv2
import math
import shutil
import numpy as np


def segment(result_path: str, yolo_path: str, new_path: str):
    ''' hyperparameter '''
    area_threshold = 30
    area_max = 10000
    
    for action_number in range(26):
        result_folder = os.path.join(result_path, 'S5', f'{action_number + 1:02d}')
        yolo_folder = os.path.join(yolo_path, f'exp{action_number + 1}') if action_number != 0 else os.path.join(yolo_path, f'exp')
        new_folder = os.path.join(new_path, 'S5', f'{action_number + 1:02d}')
        os.makedirs(new_folder, exist_ok=True)
        
        nr_image = len([name for name in os.listdir(result_folder) if name.endswith('.png')])
        for idx in range(nr_image):
            yolo_label_name = os.path.join(yolo_folder, 'labels', f'{idx}.txt')
            image_name = os.path.join(result_folder, f'{idx}.png')

            gray = cv2.imread(image_name, 0)
            output = np.zeros_like(gray)

            # step 1: calculate contours
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # step 2: Filling holes
            for contour in contours:
                cv2.drawContours(gray, [contour], 0, 255, -1)

            # step 3: Re-calculate contours
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            
            if os.path.exists(yolo_label_name):
                with open(yolo_label_name, 'r') as f:
                    data = f.read().split()
                
                H, W = gray.shape
                x_center, y_center, width, height = float(data[1]), float(data[2]), float(data[3]), float(data[4])
                
                points = [
                    [[round(x_center * W), round(y_center * H)]],
                    [[round(x_center * W), round((y_center - height / 2) * H)]],
                    [[round((x_center + width / 2) * W), round(y_center * H)]],
                    [[round(x_center * W), round((y_center + height / 2) * H)]],
                    [[round((x_center - width / 2) * W), round(y_center * H)]],
                ]

                ellipse = cv2.fitEllipse(np.array(points))

                if len(contours) == 0:
                    cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)
                    cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)

                elif len(contours) == 1:
                    area = cv2.contourArea(contours[0])
                    if area < area_threshold:
                        cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                        continue
                    
                    gray_map = gray // 255
                    orgin_pixel = gray_map.sum()

                    yolo_box = np.zeros_like(gray)
                    cv2.rectangle(yolo_box, (round((x_center - width / 2) * W), round((y_center - height / 2) * H))
                                          , (round((x_center + width / 2) * W), round((y_center + height / 2) * H))
                                          , color=1, thickness=-1)

                    intersection_pixel = (gray_map * yolo_box).sum()
                    
                    if intersection_pixel < orgin_pixel * 0.95 or orgin_pixel < area_threshold:
                        print('use yolo box: ', f'{action_number + 1:02d}-{idx:03d}')
                        cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                        continue

                    contour = cv2.convexHull(contours[0])

                    if len(contour) < 5:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue

                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)

                    orgin_pixel = (gray // 255).sum()
                    output_pixel = (output // 255).sum()
                    diff = (output_pixel + 1e-6) / (orgin_pixel + 1e-6)

                    if diff > 1.05 or diff < 0.95:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                    else:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)

                else:
                    contour_set = []
                    for i in range(len(contours)):
                        contour_set.extend(contours[i].tolist())
                    
                    contour = cv2.convexHull(np.array(contour_set))

                    area = cv2.contourArea(contour)
                    if area < area_threshold:
                        cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                        continue

                    gray_map = gray // 255
                    orgin_pixel = gray_map.sum()

                    yolo_box = np.zeros_like(gray)
                    cv2.rectangle(yolo_box, (round((x_center - width / 2) * W), round((y_center - height / 2) * H))
                                          , (round((x_center + width / 2) * W), round((y_center + height / 2) * H))
                                          , color=1, thickness=-1)

                    intersection_pixel = (gray_map * yolo_box).sum()
                    
                    if intersection_pixel < orgin_pixel * 0.95 or orgin_pixel < area_threshold:
                        print('use yolo box: ', f'{action_number + 1:02d}-{idx:03d}')
                        cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                        continue
                        
                    if len(contour) < 5:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue
                    
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)

                    orgin_pixel = (gray // 255).sum()
                    output_pixel = (output // 255).sum()
                    diff = (output_pixel + 1e-6) / (orgin_pixel + 1e-6)
                    
                    if output_pixel > area_max:
                        print(f'{output_pixel} too large!', f'{action_number + 1:02d}-{idx:03d}')
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue

                    if diff > 1.05 or diff < 0.95:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                    else:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
            else:
                if len(contours) == 0:
                    cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)

                elif len(contours) == 1:
                    area = cv2.contourArea(contours[0])
                    if area < area_threshold:
                        print('area < area_threshold:', f'{action_number + 1:02d}-{idx:03d}')
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue

                    contour = cv2.convexHull(contours[0])

                    if len(contour) < 5:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue

                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)

                    orgin_pixel = (gray // 255).sum()
                    output_pixel = (output // 255).sum()
                    diff = (output_pixel + 1e-6) / (orgin_pixel + 1e-6)

                    if diff > 1.05 or diff < 0.95:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                    else:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                
                else:
                    contour_set = []
                    for i in range(len(contours)):
                        contour_set.extend(contours[i].tolist())
                    
                    contour = cv2.convexHull(np.array(contour_set))

                    area = cv2.contourArea(contour)
                    if area < area_threshold:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue
                    
                    if len(contour) < 5:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue

                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)

                    orgin_pixel = (gray // 255).sum()
                    output_pixel = (output // 255).sum()
                    diff = (output_pixel + 1e-6) / (orgin_pixel + 1e-6)

                    if output_pixel > area_max:
                        print(f'{output_pixel} too large!', f'{action_number + 1:02d}-{idx:03d}')
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)
                        continue

                    if diff > 1.05 or diff < 0.95:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), output)
                    else:
                        cv2.imwrite(os.path.join(new_folder, f'{idx}.png'), gray)


if __name__ == '__main__':
    result_path = './Unet_drop_noTTA'
    yolo_path = '../pupil_detection/yolov5/runs/detect'
    new_path = './Unet_drop_fit_yolo'
    
    segment(result_path, yolo_path, new_path)