import os

path = './yolov5/runs/detect'
sol_path = './S5_solution'

for i in range(1, 27):
    result_folder = os.path.join(path, f'exp{i}') if i != 1 else os.path.join(path, f'exp')
    nr_image = len([name for name in os.listdir(result_folder) if name.endswith('.jpg')])

    txt_folder = os.path.join(result_folder, 'labels')
    sol_txt = os.path.join(sol_path, f'{i:02d}', 'conf.txt')

    f = open(sol_txt, 'w')
    for idx in range(nr_image):
        if os.path.exists(os.path.join(txt_folder, f'{idx}.txt')):
            print(1, file=f)
        else:
            print(0, file=f)
    
    f.close()