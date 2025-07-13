import numpy as np


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Tanh:
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dout):
        return dout * (1 - self.y ** 2)


class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

class Softmax: 
    def forward(self, x):
        x = x - np.max(x, axis=-1, keepdims=True) 
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    