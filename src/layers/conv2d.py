import numpy as np
import sys, os 
sys.path.append(os.pardir)
from utils.utils import im2col, col2im


class Conv2D:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        #中間データ
        self.x = None
        self.col = None
        self.col_W = None
        
        #重み, バイアスパラメタ勾配
        self.dW = None
        self.db = None
        
    def forward(self, x):
        N, C_in, H_in, W_in = x.shape
        FN, _, FH, FW = self.W.shape
        
        out_h = int(1 + (H_in + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W_in + 2 * self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        
        self.db = np.sum(dout, axis=0)  # (FN,)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        
        return dx