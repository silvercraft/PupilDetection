import os

count = []
for i in range(2 ** 2):
    count.append(0)

for i in range(1, 27):
    a_txt = os.path.join('remove_spike', f'{i:02d}', 'conf.txt')
    b_txt = os.path.join('ensemble_2', f'{i:02d}', 'conf.txt')

    with open(a_txt, 'r') as f:
        a_conf = f.read().splitlines()
    with open(b_txt, 'r') as f:
        b_conf = f.read().splitlines()
    
    for j in range(len(a_conf)):
        bit_1 = int(a_conf[j])
        bit_2 = int(b_conf[j])

        idx = bit_1 * 2 + bit_2 * 1
        count[idx] += 1

print(count)