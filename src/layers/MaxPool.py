import numpy as np
import sys, os 
sys.path.append(os.pardir)
from utils.utils import im2col, col2im

class MaxPool:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # 中間データ
        self.x_shape = None
        self.col = None
        self.col_argmax = None

    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - self.pool_h) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.col = col
        self.col_argmax = np.argmax(col, axis=1)

        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout_flat = dout.transpose(0, 2, 3, 1).flatten()

        dcol = np.zeros_like(self.col)
        dcol[np.arange(len(self.col)), self.col_argmax] = dout_flat

        dcol = dcol.reshape(self.col.shape)
        dx = col2im(dcol, self.x_shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx
