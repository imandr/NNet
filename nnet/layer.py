import numpy as np

class Layer(object):

    Index = 0
    
    def __init__(self, name=None, lid=None):
        if name is None:
            Layer.Index += 1
            name = "{}_{}".format(self.__class__.__name__, self.Index)
        self.ID = lid or id(self)      # used by get_config
        self.Name = name
        self.Inputs = []
        self.Shape = None       # will be calculated by init()
        self.Tick = None
        self.Xs = None
        self.Y = None
        self.Gp = None
        self.Gx = None
        self.State = None
        self.State1 = None
        self.TrainT = None
        self.TrainerData = None
        self.Params = []
        self.Regularizable = ()

    def newTick(self, t):
        self.Y = None
        self.Xs = None
        self.Gp = None
        self.Gx = None
        self.State = self.State1
        self.Tick = t
        self.tick()
        
    def resetState(self, t):
        if t != self.Tick:
            self.newTick(t)
            self.State = None
            self.reset_state()
            for inp in self.Inputs:
                inp.resetState(t)
        
    def link(self, inputs):
        if isinstance(inputs, Layer):
            inputs = [inputs]
        elif isinstance(input, list):
            for x in inputs:
                assert isinstance(x, Layer)
        self.Inputs = inputs        # list of layers
        self.init(inputs)
        
    def __call__(self, *inputs):
        self.link(inputs)
        return self
        
    def forward(self, t):
        if t != self.Tick and not t is None:
            self.newTick(t)
        if self.Y is None:
            inp_x = []
            for inp in self.Inputs:
                inp_x.append(inp.forward(t))
            self.Xs = inp_x
            y, state1 = self.FP(inp_x, self.State)
            #print "%s.forward: inputs: " % (self.Name,), [x.shape for x in inp_x], "   -> y:", y.shape
            self.Y = y
            self.State1 = state1
        return self.Y

    def addList(self, t0, t1):
        #print "%s: addList(%s, %s)" % (self.Name, t0, t1)
        if t0 is None:
            return t1
        else:
            for x0, x1 in zip(t0, t1):
                x0 += x1
            return t0
    
    def backward(self, gy, t):
        assert t == self.Tick
        gxs, gp = self.BP(gy)
        #print "%s.backward: gy: %s     ->" % (self.Name, gy.shape), \
        #        "gx:", [gx.shape for gx in gxs], \
        #        "gp:", [gpi.shape for gpi in gp]
        #print self.Name, "gy=",gy
        self.Gp = self.addList(self.Gp, gp)
        #print self.Name, "Gp=",self.Gp
        for inp, gx in zip(self.Inputs, gxs):
            inp.backward(gx, t)
    
    def grads(self):
        return self.Gp

    def train(self, trainer, regularizer, t):
        if t != self.TrainT:
            self.TrainT = t
            if self.Params:
                #print self.Name, "params=    ",self.Params
                #print self.Name, "    Gp=    ",self.Gp
                self.TrainerData = trainer.apply(self.Params, self.Gp, self.TrainerData)
                #print self.Name, "new params=",self.Params
                if regularizer is not None:
                    regularized = regularizer.apply(self.Regularizable)
            for inp in self.Inputs:
                inp.train(trainer, regularizer, t)
                
    @staticmethod
    def from_config(cfg):
        if cfg["type"] in ["linear", "merge", "flatten", "input"]:
            from .core_layers import core_layer
            return core_layer(cfg)
            
        elif cfg["type"] == "loss":
            from .losses import LossLayer
            return LossLayer.from_config(cfg)
            
        elif cfg["type"] == "activation":
            from .activations import ActivationLayer
            return ActivationLayer.from_config(cfg)
                
#
# overridables
#
    def init(self, inputs):
        pass
        
    def tick(self):
        pass
        
    def reset_state(self):
        pass
        
    def FP(self, x, s):
        raise NotImplementedError("FP method not implemented for layer %s %s" % (self.Name, self.__class__.__name__))
        
    def BP(self, gy):
        raise NotImplementedError
        
    def config(self):
        return dict(id = self.ID, inputs = [l.ID for l in self.Inputs or []], name=self.Name or "")
                
    def nparams(self):
        return len(self.Params) if self.Params else 0
        
    def get_params(self):
        return self.Params or []
        
    def get_grads(self):
        return self.Gp or []
        
    def set_params(self, params):
        assert len(params) == len(self.Params)
        for p, q in zip(self.Params, params):
            assert p.shape == q.shape
            p[...] = q
    