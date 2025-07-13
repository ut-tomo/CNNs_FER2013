import numpy as np
import sys, os 
sys.path.append(os.pardir)
from utils.utils import im2col, col2im

class AvePool:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        self.x_shape = x.shape  # for backward
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - self.pool_h) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - self.pool_w) / self.stride)

        self.col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = self.col.reshape(-1, self.pool_h * self.pool_w)

        out = np.mean(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout_flat = dout.transpose(0, 2, 3, 1).flatten()

        pool_size = self.pool_h * self.pool_w
        dcol = np.repeat(dout_flat[:, np.newaxis], pool_size, axis=1)
        dcol /= pool_size

        dcol = dcol.reshape(self.col.shape)
        dx = col2im(dcol, self.x_shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx