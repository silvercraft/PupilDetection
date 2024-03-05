import os
import sys
import json

dir_path = sys.argv[1]

f = open(sys.argv[2])
confs = json.load(f)
nm = os.path.split(sys.argv[2])[1].split(".")[0]

for dir in sorted(os.listdir(dir_path)):
    conf = confs["result"]

    path = os.path.join(dir_path, f'{nm}.txt')
    f = open(path, 'w')
    for i, c in enumerate(conf):
        c = str(c)
        if i == 0:
            f.write(c)
            # f.write('1.0')
        else:
            f.write(f'\n{c}')
            # f.write('\n1.0')
    f.close()
