import os
import cv2

'''
S1 - 05
'''

# for i in range(1, 27):
#     print(f'{i = }')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(f'S5_video/S5_{i:02d}.mp4', fourcc, 10.0, (640, 480))

#     img_list = os.listdir(f'./dataset/public/S5/{i:02d}')
#     img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0]))

#     for filename in img_list:
#         img = cv2.imread(os.path.join(f'./dataset/public/S5/{i:02d}', filename))
#         out.write(img)

#     out.release()

# for i in range(1, 27):
#     print(f'{i = }')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(f'./test_video/opencv_S1_{i:02d}.mp4', fourcc, 5.0, (640, 480))

#     img_list = [fn for fn in os.listdir(f'./test/S1/{i:02d}') if fn.endswith('draw.png')]
#     img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0][:-5]))

#     for filename in img_list:
#         img = cv2.imread(os.path.join(f'./test/S1/{i:02d}', filename))
#         out.write(img)

#     out.release()


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f'./ensemble_public_fit.mp4', fourcc, 20.0, (640, 480))

# dir = f'./trash/pupil_detection/yolov5/runs/detect/exp'
# img_list = [fn for fn in os.listdir(dir) if fn.endswith('.jpg')]

dir = f'./reproduce/ensemble_public_fit/S5_solution/03'
img_list = [fn for fn in os.listdir(dir) if fn.endswith('.png')]

img_list = sorted(img_list, key=lambda x: int(os.path.splitext(x)[0]))

for filename in img_list:
    img = cv2.imread(os.path.join(dir, filename))
    out.write(img)

out.release()