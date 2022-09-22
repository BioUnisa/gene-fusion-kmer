import numpy as np

def one_hot_encode(seq):
    map = np.asarray([
        [0, 0, 0, 0],   # N
        [1, 0, 0, 0],   # A
        [0, 1, 0, 0],   # C
        [0, 0, 1, 0],   # G
        [0, 0, 0, 1]    # T
    ])
    seq = seq.upper().replace('A', '\x01', ).replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')
    return map[np.frombuffer(bytes(seq, 'utf-8'), np.int8) % 5]