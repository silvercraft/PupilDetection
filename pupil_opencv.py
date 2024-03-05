import os
import cv2
import math
import shutil
import numpy as np

from tqdm import tqdm


def segment(dataset_path: str, result_path: str, subjects: list):
    ''' hyperparameter '''
    pixel_value_threshold = 30
    area_threshold = 300
    extend_threshold = 0.8
    circularity_threshold = 2
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


    sequence_idx = 0
    for subject in subjects:
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            result_folder = os.path.join(result_path, subject, f'{action_number + 1:02d}')
            os.makedirs(result_folder)
            
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                image = cv2.imread(image_name)
                # cv2.imshow("image", image)
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                retval, thresholded = cv2.threshold(gray, pixel_value_threshold, 255, 0)
                # cv2.imshow("threshold", thresholded)

                closing = cv2.erode(cv2.dilate(thresholded, kernel, iterations=1), kernel, iterations=1)
                # closing = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel)
                # cv2.imshow("closing", closing)

                contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                drawing = np.copy(image)
                output = np.zeros_like(gray)
                # cv2.drawContours(drawing, contours, -1, (255, 0, 0), 2)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    bounding_box = cv2.boundingRect(contour)

                    # method 1: remove small noise
                    if area < area_threshold:
                        continue

                    # bounding_box: x, y, w, h
                    extend = area / (bounding_box[2] * bounding_box[3])

                    # method 2: remove the contours with big extend
                    if extend > extend_threshold:
                        continue
                    
                    # method 3: improving roundness
                    contour = cv2.convexHull(contour)

                    circumference = cv2.arcLength(contour, True)
                    circularity = circumference ** 2 / (4 * math.pi * area)

                    # method 4: remove the circularity > 2
                    if circularity > circularity_threshold:
                        continue

                    # calculate countour center and draw a dot there
                    m = cv2.moments(contour)
                    if m['m00'] != 0:
                        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
                        cv2.circle(drawing, center, 3, (0, 255, 0), -1)

                    # fit an ellipse around the contour and draw it into the image
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        cv2.ellipse(drawing, box=ellipse, color=(0, 255, 0))
                        cv2.ellipse(output, box=ellipse, color=(255, 255, 255), thickness=-1)
                    except:
                        print('error')
                        pass
                
                # cv2.imshow("Drawing", drawing)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                cv2.imwrite(os.path.join(result_folder, f'{idx}_draw.png'), drawing)
                cv2.imwrite(os.path.join(result_folder, f'{idx}.png'), output)


if __name__ == '__main__':
    dataset_path = './dataset/public'
    result_path = './test'

    # subjects = ['S1', 'S2', 'S3', 'S4']
    subjects = ['S5']

    # shutil.rmtree(result_path)
    segment(dataset_path, result_path, subjects)