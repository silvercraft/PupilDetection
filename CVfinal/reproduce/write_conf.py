import os, sys
import json

dir_path = sys.argv[1]

f = open(sys.argv[2])
confs = json.load(f)

for dir in sorted(os.listdir(dir_path)):
    conf = confs[dir]
    number = len(os.listdir(os.path.join(dir_path, dir)))
    print(len(conf), number)
    path = os.path.join(dir_path, dir, 'conf.txt')
    f = open(path, 'w')
    for i, c in enumerate(conf):
        c = str(c)
        if i==0:
            f.write(c)
            # f.write('1.0')
        else:
            f.write(f'\n{c}')
            # f.write('\n1.0')
    f.close()