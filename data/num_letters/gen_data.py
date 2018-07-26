import argparse
import string
import sys
import random

LEN = int(sys.argv[1])
SIZE = int(sys.argv[2])

letters = list(string.ascii_uppercase)

data = []
maxnum = len(letters)
for j in range(SIZE):
    # With replacement
    x = [random.choice(range(maxnum)) for i in range(LEN)]
    y_num = sorted(x)
    y_let = [letters[i] for i in y_num]
    x = [str(i) for i in x]
    y_num = [str(i) for i in y_num]
    data.append((x, y_num, y_let))

DEV_SPLIT = int(SIZE * 0.8)
TEST_SPLIT = int(SIZE * 0.9)

train_data = data[:DEV_SPLIT]
dev_data = data[DEV_SPLIT:TEST_SPLIT]
test_data = data[TEST_SPLIT:]

def record_data(split, data):
    with open(split + '/input.txt', 'w') as f1, open(split + '/output_num.txt', 'w') as f2, open(split + '/output_let.txt', 'w') as f3:
        for inst in data:
            f1.write(' '.join(inst[0]) + '\n')
            f2.write(' '.join(inst[1]) + '\n')
            f3.write(' '.join(inst[2]) + '\n')
            
            
record_data('train', train_data)
record_data('dev', dev_data)
record_data('test', test_data)
