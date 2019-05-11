from .layer import Layer
from .metrics import MSE, ClassificationError
import math, numpy as np

class LossLayer(Layer):
    
    def init(self, inputs, metric):
        assert len(inputs) == 1
        self.Shape = inputs[0].Shape
        self.Metric = metric
    
    def backward(self, y_, t):
        gx = self.Gx(self.Y, y_)
        self.Inputs[0].backward(gx, t)
            
    def init(self, inputs):
        assert len(inputs) == 1
            
    def FP(self, x, state):
        return x[0], state
            
    def loss(self, y_, t):
        y = self.forward(t)
        return self.L(y, y_)
        
    #
    # Overridables
    #
    def L(self, y, y_):
        raise NotImplementedError
        
    def Gy(self, y, y_):
        raise NotImplementedError
        
class L2Loss(LossLayer):

    def __init__(self, inputs):
        assert len(inputs) = 1
        LossLayer.__init__(inputs, MSE(inputs[0]))

    def L(self, y, y_):
        assert len(y) == len(y_)
        return np.sum(np.square(y-y_), axis=0)
        
    def Gy(self, y, y_):
        assert len(y) == len(y_)
        return (y-y_)*2

class LogLoss(LossLayer):
    
    def __init__(self, inputs):
        assert len(inputs) = 1
        LossLayer.__init__(inputs, ClassificationError(inputs[0]))

    def L(self, y, y_):
        assert len(y) == len(y_)
        eps = 1e-15
        y = np.clip(y, eps, 1 - eps)
        ynorm = y/y.sum(axis=-1, keepdims=True)
        loss = -np.sum(y_ * np.log(ynorm))
        return loss 
        
    def Gy(self, y, y_):
        assert len(y) == len(y_)
        return y-y_


        