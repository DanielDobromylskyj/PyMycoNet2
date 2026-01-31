import numpy as np

import pymyconet2
from pymyconet2.layers.dense import DenseLayer
from pymyconet2.layers.convoluted import Convoluted
from pymyconet2.layers.transform import Condense
from pymyconet2.activation import tanh, softmax

net = pymyconet2.Network((
    Convoluted((2, 2), 1, 4),
    Condense(),
    tanh.TanH(),
    DenseLayer(64, 2),
    softmax.SoftMax()
))

input_data = [
    np.array([
        [0, 1, 1, 0],
        [0, 1, 1 ,0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=np.float32),
    np.array([
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32)
]

targets = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

def log():
    for i in range(len(input_data)):
        output = net.forward_single(input_data[i])
        print(f"Test {i}: Target: {[round(x, 3) for x in targets[i]]} Output: {[round(x, 3) for x in output]}")


print("\nPre Training")
log()
print()

net.train(input_data, targets, epoches=200, max_batch_size=1, learning_rate=0.1)


print("\nAfter Training")
log()
