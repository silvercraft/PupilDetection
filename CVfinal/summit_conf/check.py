import os
import cv2
import numpy as np

for i in range(1, 27):
    if i != 16:
        continue
    a_txt = os.path.join('./ensemble_2', f'{i:02d}', 'conf.txt')

    with open(a_txt, 'r') as f:
        a_conf = f.read().splitlines()

    for j in range(len(a_conf)):
        print(f'{j} - {a_conf[j]}')