from .layer import Layer
import math, numpy as np

class ActivationLayer(Layer):
    
    def init(self, inputs):
        assert len(inputs) == 1
        self.Shape = inputs[0].Shape
    
    def FP(self, x, state):
        assert len(x) == 1
        y = self.Y(x[0])
        return y, None
        
    def BP(self, gy):
        assert len(gy) == 1
        return [self.Gx(self.X[0], self.Y)*gy], None
        
    #
    # Overridables
    #
    def Y(self, x):
        raise NotImplementedError
        
    def Gx(self, x, y):
        raise NotImplementedError

class Tanh(ActivationLayer): 
    
    def Y(self, x):
        return np.tanh(x)
        
    def Gx(self, x, y):
        return 1.0 - np.square(y)
        
class Sigmoid(ActivationLayer):
    
    def Y(self, x):
        return np.tanh(x)/2 + 1
        
    def Gx(self, x, y):
        return y*(1-y)
        
class SoftPlus(ActivationLayer):
    
    def Y(self, x):
        return np.log(np.exp(x)+1.0)

    def Gx(self, x, y):
        e = np.exp(x)
        return e/(1.0+e)
        
class ReLU(ActivationLayer):
    
    def Y(self, x):
        return (x + np.abs(x))/2
        
    def Gx(self, x, y):
        gx = np.zeros_like(x)
        gx[x >= 0] = 1.0
        return gx
        

        
               
        