import os, sys
import json

dir_path = sys.argv[1]

f = open(sys.argv[2])
confs = json.load(f)

for sdir in sorted(os.listdir(dir_path)):
    for dir in sorted(os.listdir(os.path.join(dir_path, sdir))):
        conf = confs[os.path.join(sdir, dir)]
        number = len(os.listdir(os.path.join(dir_path, sdir, dir)))
        print(len(conf), number)
        path = os.path.join(dir_path, sdir, dir, 'conf.txt')
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