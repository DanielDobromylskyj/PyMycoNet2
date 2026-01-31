import numpy as np

import pymyconet2
from pymyconet2.layers.dense import DenseLayer
from pymyconet2.layers.convoluted import Convoluted
from pymyconet2.layers.transform import Condense
from pymyconet2.activation import tanh, softmax

net = pymyconet2.Network((
    Convoluted((2, 2), 2, 4),
    Condense(),
    tanh.TanH(),
    DenseLayer(16, 2),
    softmax.SoftMax()
))

input_data = np.array([
    [1, 1,  -1,  -1],
    [1, 1,  -1,  -1],
    [0, 0, 0.5, 0.5],
    [0, 0, 0.5, 0.5],
], dtype=np.float32)

output = net.forward_single(input_data)

print(output)