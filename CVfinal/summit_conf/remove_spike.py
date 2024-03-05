import os

for i in range(1, 27):
    a_txt = os.path.join('./ensemble_2', f'{i:02d}', 'conf.txt')

    with open(a_txt, 'r') as f:
        a_conf = f.read().splitlines()
    
    os.mkdir(os.path.join('./remove_spike', f'{i:02d}'))
    new_txt = os.path.join('./remove_spike', f'{i:02d}', 'conf.txt')
    new = open(new_txt, 'w')

    print(a_conf[0], file=new)
    print(a_conf[1], file=new)
    
    for j in range(2, len(a_conf) - 5):
        sub = [a_conf[j-2], a_conf[j-1], a_conf[j], a_conf[j+1], a_conf[j+2]]
        if sub[0] == '0' and sub[1] == '0' and sub[2] == '1' and sub[3] == '0' and sub[4] == '0':
            print(f'{i}-{j}')
            print(0, file=new)
        else:
            print(a_conf[j], file=new)
    
    for j in range(len(a_conf) - 5, len(a_conf)):
        print(a_conf[j], file=new)
    
    new.close()