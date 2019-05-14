from .layer import Layer
import numpy as np
import math

class Linear(Layer):
    
    def __init__(self, nout, name=None, lid=None):
        Layer.__init__(self, name=name, lid=lid)
        self.NOut = nout
        
    def init(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        self.NIn = inp.Shape[-1]
        W_shape = (self.NIn, self.NOut)
        rng = np.random.RandomState()
        #self.W = rng.normal(size=W_shape, scale=math.sqrt(2.0/(self.NIn + self.NOut)))
        self.W = (np.random.random(W_shape)*2 - 1) * math.sqrt(6.0/(self.NIn + self.NOut))
        self.b = np.zeros(self.NOut)
        self.Shape = inp.Shape[:-1] + (self.NOut,)
        self.Params = (self.W, self.b)
        self.Regularizable = (self.W,)

    def FP(self, xs, state):
        assert len(xs) == 1
        x = xs[0]
        y = np.dot(x, self.W) + self.b
        return y, state
        
    def BP(self, gy):
        #print "linear.bprop: x:", x[0].shape, "   gy:", gY.shape
        x = self.Xs[0]
        n_mb = len(x)
        inp_flat = x.reshape((-1, self.NIn))
        g_flat = gy.reshape((-1, self.NOut))
        gW = np.dot(inp_flat.T, g_flat)/n_mb    # [...,n_in],[...,n_out]
        gb = np.mean(g_flat, axis=0)
        gx = np.dot(gy, self.W.T)
        return [gx], [gW, gb]
        
    def config(self):
        cfg = Layer.config(self)
        cfg["type"] = "linear"
        cfg["nout"] = self.NOut
        return cfg

class Merge(Layer):
    
    def __init__(self, axis=0, name=None, lid=None):
        Layer.__init__(self, name=name, lid=lid)
        assert axis == 0        # for now , 0 is next axis after the one along the minibatch
        self.Axis = axis

    def init(self, inputs):
        shape0 = inputs[0].shape()
        shape0[self.Axis] = 0
        n_total = 0
        for inp in inputs:
            shape = inp.Shape
            n_total += shape[self.Axis]
            shape[self.Axis] = 0
            assert shape == shape0
        self.Shape = [n_total]+shape0[1:]
        self.Params = ()
        
    def FP(self, xs, state):
        return np.concatenate(xs, axis=self.Axis), state
        
    def BP(self, gy):
        i = 0
        gxs = []
        for inp in self.Inputs:
            gxs.append(gy[:,i+inp.shape()[0]])
        return gxs, None
        
    def config(self):
        cfg = Layer.config(self)
        cfg["type"] = "merge"
        cfg["axis"] = self.Axis
        return cfg

class Input(Layer):
    
    def __init__(self, shape, name=None, lid=None):
        Layer.__init__(self, name=name, lid=lid)
        self.Shape = shape
        
    def set(self, value, t):
        self.newTick(t)
        self.Xs = [value]
        
    def forward(self, t=None):
        assert t == self.Tick
        self.Y = self.Xs[0]
        return self.Y
        
    def BP(self, gy):
        return (), ()

    def config(self):
        cfg = Layer.config(self)
        cfg["type"] = "input"
        cfg["shape"] = self.Shape
        return cfg
        
class Flatten(Layer):
    
    def init(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        self.InShape = inp.Shape
        n = 1
        for x in self.InShape:  n *= x
        self.Shape = (n,)
        
    def FP(self, xs, state):
        assert len(xs) == 1
        x = xs[0]
        return x.reshape((x.shape[0], -1)), state
        
    def BP(self, gy):
        return [gy.reshape((gy.shape[0],)+self.InShape)], []

    def config(self):
        cfg = Layer.config(self)
        cfg["type"] = "flatten"
        return cfg
        
def core_layer(cfg):
    if cfg["type"] == "flatten":
        return Flatten(name=cfg.get("name"), lid=cfg["id"])
    elif cfg["type"] == "input":
        return Input(tuple(cfg["shape"]), name=cfg.get("name"), lid=cfg["id"])
    elif cfg["type"] == "merge":
        return Merge(axis=cfg["axis"], name=cfg.get("name"), lid=cfg["id"])
    elif cfg["type"] == "linear":
        return Linear(cfg["nout"], name=cfg.get("name"), lid=cfg["id"])
    else:
        raise ValueError("Unknown core layer type: %s" % (cfg["type"],))
 