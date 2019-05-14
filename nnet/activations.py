from .layer import Layer
import math, numpy as np

class ActivationLayer(Layer):
    
    def init(self, inputs):
        assert len(inputs) == 1
        self.Shape = inputs[0].Shape
    
    def FP(self, x, state):
        assert len(x) == 1
        y = self.f(x[0])
        return y, None
        
    def BP(self, gy):
        return [self.gx(self.Xs[0], self.Y)*gy], []
        
    #
    # Overridables
    #
    def f(self, x):
        raise NotImplementedError
        
    def gx(self, x, y):
        # dy/dx, y is given for convenience
        raise NotImplementedError
        
    def config(self):
        cfg = Layer.config(self)
        cfg["type"] = "activation"
        return cfg

    @staticmethod
    def from_config(cfg):
        assert cfg["type"] == "activation"
        if cfg["function"] == "tanh":
            return Tanh(name=cfg.get("name"), lid=cfg["id"])
        elif cfg["function"] == "sigmoid":
            return Sigmoid(name=cfg.get("name"), lid=cfg["id"])
        elif cfg["function"] == "softplus":
            return Softplus(name=cfg.get("name"), lid=cfg["id"])
        elif cfg["function"] == "relu":
            return ReLU(name=cfg.get("name"), lid=cfg["id"])
        else:
            raise ValueError("Unknown activation function: "+cfg["function"])
        
class Tanh(ActivationLayer): 
    
    def f(self, x):
        return np.tanh(x)
        
    def gx(self, x, y):
        return 1.0 - np.square(y)

    def config(self):
        cfg = ActivationLayer.config(self)
        cfg["function"] = "tanh"
        return cfg

        
class Sigmoid(ActivationLayer):
    
    def f(self, x):
        return np.tanh(x)/2 + 1
        
    def gx(self, x, y):
        return y*(1-y)
        
    def config(self):
        cfg = ActivationLayer.config(self)
        cfg["function"] = "sigmoid"
        return cfg

class SoftPlus(ActivationLayer):
    
    def f(self, x):
        return np.log(np.exp(x)+1.0)

    def gx(self, x, y):
        e = np.exp(x)
        return e/(1.0+e)

    def config(self):
        cfg = ActivationLayer.config(self)
        cfg["function"] = "softplus"
        return cfg
        
class ReLU(ActivationLayer):
    
    def f(self, x):
        return (x + np.abs(x))/2
        
    def gx(self, x, y):
        gx = np.zeros_like(x)
        gx[x >= 0] = 1.0
        return gx
        
    def config(self):
        cfg = ActivationLayer.config(self)
        cfg["function"] = "relu"
        return cfg

        
               
        