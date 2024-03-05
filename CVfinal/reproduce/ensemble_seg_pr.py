import numpy as np
import cv2
import os
import sys

paths = [sys.argv[1], sys.argv[2], sys.argv[3]]
file_name = []
imgs = []

for i,path in enumerate(paths):
    idx = 0
    for spath in sorted(os.listdir(path)):
        for dir in sorted(os.listdir(os.path.join(path, spath))):
            for j, file in enumerate(sorted(os.listdir(os.path.join(path, spath, dir)))):
                if file[-3:]=='txt':
                    continue
                file_name.append(os.path.join(spath, dir, file))
                img = np.array(cv2.imread(os.path.join(path, spath, dir, file)), np.int32)
                if i==0:
                    imgs.append(img)
                else:
                    imgs[idx] += img
                    idx += 1
                    # print(imgs[j].max())

print(np.array(imgs).shape)

imgs = np.array(imgs)
for i in range(imgs.shape[0]):
    print(f'thresh: {255*((len(paths))//2 + 1)}')
    white_index = imgs[i] >= 255*((len(paths))//2 + 1)
    black_index = imgs[i] < 255*((len(paths))//2 + 1)
    imgs[i][white_index] = 255
    imgs[i][black_index] = 0

    print(file_name[i])
    os.makedirs(os.path.join(sys.argv[4], file_name[i].split('/')[0], file_name[i].split('/')[1]), exist_ok = True)
    cv2.imwrite(os.path.join(sys.argv[4], file_name[i]), imgs[i])