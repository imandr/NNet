from .layer import Layer
from .metrics import MSE, ClassificationError
import math, numpy as np
from .core_layers import Flatten, Linear

class LossLayer(Layer):
    
    def __init__(self, metric=None, name=None, lid=None):
        Layer.__init__(self, name=name, lid=lid)
        self.Metric = metric          
    
    def backward(self, y_, t):
        assert t == self.Tick
        gxs = self.grads(self.Y, y_)
        for inp, gx in zip(self.Inputs, gxs):
            inp.backward(gx, t)

    @staticmethod
    def from_config(cfg):
        assert cfg["type"] == "loss"
        if cfg["function"] == "l2":
            return L2Loss(name=cfg.get("name"), lid=cfg["id"])
        elif cfg["function"] == "crossentropy":
            return CrossEntoryLoss(name=cfg.get("name"), lid=cfg["id"])
        else:
            raise ValueError("Unknown loss function: "+cfg["function"])

    #
    # Overridables
    #
    def loss(self, y, y_):
        raise NotImplementedError
        
    def grads(self, y, y_):
        raise NotImplementedError
        
    def config(self):
        cfg = Layer.config(self)
        cfg["type"] = "loss"
        return cfg
    
        
class L2Loss(LossLayer):
    
    def __init__(self, name=None, lid=None):
        LossLayer.__init__(self, metric = MSE(), name=name, lid=lid)

    def init(self, inputs):
        assert len(inputs) == 1
        assert len(inputs[0].Shape) == 1
        self.Shape = inputs[0].Shape
        
    def FP(self, xs, state):
        return xs[0], state
        
    def loss(self, y_):
        y = self.Y
        assert len(y) == len(y_)
        return np.mean(np.sum(np.square(y-y_), axis=-1), axis=0)
        
    def grads(self, y, y_):
        assert len(y) == len(y_)
        return [(y-y_)*2]
        
    def config(self):
        cfg = LossLayer.config(self)
        cfg["function"] = "l2"
        return cfg

class CrossEntropyLoss(LossLayer):
    
    # input is unnormalized values before x -> exp(x)/sum(exp(x))
    
    def __init__(self, name=None, lid=None):
        #print "creating LogLoss"
        LossLayer.__init__(self, metric = ClassificationError(), name=name, lid=lid)

    def init(self, inputs):
        assert len(inputs) == 1
        assert len(inputs[0].Shape) == 1
        assert isinstance(inputs[0], Linear), "Input for CrossEntropy loss should be a linear layer"
        self.Shape = inputs[0].Shape
        
    def loss(self, y_):
        y = self.Y
        assert len(y) == len(y_)
        eps = 1e-15
        y = np.clip(y, eps, 1 - eps)
        loss = -np.mean(np.sum(y_ * np.log(y), axis=-1), axis=0)
        return loss 

    def FP(self, xs, state):
        x = xs[0]
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        y = e/np.sum(e, axis=-1, keepdims=True)
        return y, state
        
    def grads(self, y, y_):
        #print "softmax.grads: y=",y,"   y_=",y_
        assert len(y) == len(y_)
        return [y-y_]
        
    def config(self):
        cfg = LossLayer.config(self)
        cfg["function"] = "crossentropy"
        return cfg
