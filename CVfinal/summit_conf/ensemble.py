import os


for i in range(1, 27):
    a_txt = os.path.join('v2', f'{i:02d}', 'conf.txt')
    b_txt = os.path.join('resnet_20', f'{i:02d}', 'conf.txt')
    c_txt = os.path.join('conf0613', f'{i:02d}', 'conf.txt')

    with open(a_txt, 'r') as f:
        a_conf = f.read().splitlines()
    with open(b_txt, 'r') as f:
        b_conf = f.read().splitlines()
    with open(c_txt, 'r') as f:
        c_conf = f.read().splitlines()
    
    os.mkdir(os.path.join('./ensemble_2', f'{i:02d}'))
    new_txt = os.path.join('./ensemble_2', f'{i:02d}', 'conf.txt')
    new = open(new_txt, 'w')
    
    for j in range(len(a_conf)):
        bit_1 = int(a_conf[j])
        bit_2 = int(b_conf[j])
        bit_3 = int(c_conf[j])

        ''' ensemble 2 '''
        total = 4 * bit_1 + 2 * bit_2 + bit_3
        if total > 4:
            print(1, file=new)
        else:
            print(0, file=new)
    
    new.close()