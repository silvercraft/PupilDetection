import os
import cv2

count = []
for i in range(2 ** 2):
    count.append(0)

yolo_path = '../../pupil_detection/yolov5/runs/detect'
seg_path = '../0610_s5b4'
a = 0

for i in range(1, 27):
    a_txt = os.path.join('ensemble_2', f'{i:02d}', 'conf.txt')
    b_txt = os.path.join('yolo', f'{i:02d}', 'conf.txt')

    with open(a_txt, 'r') as f:
        a_conf = f.read().splitlines()
    with open(b_txt, 'r') as f:
        b_conf = f.read().splitlines()
    
    os.mkdir(os.path.join('./test', f'{i:02d}'))
    new_txt = os.path.join('./test', f'{i:02d}', 'conf.txt')
    new = open(new_txt, 'w')
    
    for j in range(len(a_conf)):
        bit_1 = int(a_conf[j])
        bit_2 = int(b_conf[j])

        idx = bit_1 * 2 + bit_2 * 1
        count[idx] += 1

        if idx == 1:
            result_folder = os.path.join(seg_path, 'S5', f'{i:02d}')
            image_name = os.path.join(result_folder, f'{j}.png')
            gray = cv2.imread(image_name, 0)

            yolo_folder = os.path.join(yolo_path, f'exp{i}') if i != 1 else os.path.join(yolo_path, f'exp')
            yolo_label_name = os.path.join(yolo_folder, 'labels', f'{j}.txt')
            with open(yolo_label_name, 'r') as f:
                data = f.read().split()

            H, W = gray.shape
            x_center, y_center, width, height = float(data[1]), float(data[2]), float(data[3]), float(data[4])
            
            top_left_x = x_center - width / 2
            top_left_y = y_center - height / 2
            bottom_right_x = x_center + width / 2
            bottom_right_y = y_center + height / 2
            top_left_min = min(top_left_x, top_left_y)
            bottom_right_max = max(bottom_right_x, bottom_right_y)

            if top_left_min < 0.01 or bottom_right_max > 0.99:
                print(0, file=new)
                continue
            
            a+=1
            print(f'{i}-{j}')
            print(1, file=new)
        
        else:
            print(bit_1, file=new)
    
    new.close()

print(a)