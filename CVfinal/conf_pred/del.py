import os

path = 'S5_solution'

for action_number in range(26):
    folder = os.path.join(path, f'{action_number + 1:02d}')
    nr_image = len([name for name in os.listdir(folder) if name.endswith('.png')])

    for idx in range(nr_image):
        image_name = os.path.join(folder, f'{idx}.png')
        os.remove(image_name)

    # conf_txt = os.path.join(path, f'{action_number + 1:02d}', 'conf.txt')
    # os.remove(conf_txt)