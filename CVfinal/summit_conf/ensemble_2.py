import os


for i in range(1, 27):
    a_txt = os.path.join('yolo', f'{i:02d}', 'conf.txt')
    b_txt = os.path.join('ensemble_2', f'{i:02d}', 'conf.txt')

    with open(a_txt, 'r') as f:
        a_conf = f.read().splitlines()
    with open(b_txt, 'r') as f:
        b_conf = f.read().splitlines()
    

    os.mkdir(os.path.join('ensemble_3', f'{i:02d}'))
    new_txt = os.path.join('ensemble_3', f'{i:02d}', 'conf.txt')
    new = open(new_txt, 'w')
    
    for j in range(len(a_conf)):
        bit_1 = int(a_conf[j])
        bit_2 = int(b_conf[j])

        ''' ensemble '''
        total = 2 * bit_1 + bit_2
        if total < 3:
            print(0, file=new)
        else:
            print(1, file=new)
    
    new.close()