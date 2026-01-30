import numpy as np
import pymyconet2
from pymyconet2.layers.dense import DenseLayer

# XOR network
net = pymyconet2.Network((
    DenseLayer(2, 8, pymyconet2.TANH),
    DenseLayer(8, 8, pymyconet2.TANH),
    DenseLayer(8, 2, pymyconet2.SOFTMAX)
))

# Training data: XOR
training_inputs = [
    np.array([0.0, 0.0], dtype=np.float32),
    np.array([0.0, 1.0], dtype=np.float32),
    np.array([1.0, 0.0], dtype=np.float32),
    np.array([1.0, 1.0], dtype=np.float32),
]

training_targets = [
    np.array([1, 0], dtype=np.float32),
    np.array([0, 1], dtype=np.float32),
    np.array([0, 1], dtype=np.float32),
    np.array([1, 0], dtype=np.float32),
]

net.train(
    train_inputs=training_inputs,
    train_targets=training_targets,
    epoches=2000,       # more epochs to ensure convergence
    max_batch_size=1,   # batch size = 1 for stochastic updates
    learning_rate=0.1   # bigger LR helps tiny network converge
)

# Test the network after training
for x in training_inputs:
    out = net.forward_single(x)
    print(f"Input: {x}, Output: {np.asarray([round(y, 4) for y in out])}")
