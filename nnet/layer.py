import numpy as np

class Layer(object):

    Index = 0
    
    def __init__(self, name=None):
        if name is None:
            Layer.Index += 1
            name = "{}_{}".format(self.__class__.__name__, self.Index)
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
        self.Params = None
        self.Regularizable = ()

    def newTick(self, t):
        self.Y = None
        self.X = None
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
        self.Inputs = inputs        # list of layers
        self.init(inputs)
        
    def __call__(self, *inputs):
        self.link(inputs)
        return self
        
    def forward(self, t=None):
        if t != self.Tick and not t is None:
            self.newTick(t)
        if self.Y is None:
            inp_x = []
            for inp in self.Inputs:
                inp_x += inp.forward(t)
            self.Xs = inp_x
            y, state1 = self.FP(inp_x, self.State)
            self.Y = y
            self.State1 = state1
        return self.Y

    def addList(self, t0, t1):
        if t0 is None:
            return t1
        else:
            for x0, x1 in zip(t0, t1):
                x0 += x1
            return t0
    
    def backward(self, gy, t):
        assert t == self.Tick
        gxs, gp = self.BP(gy)
        self.Gp = self.addList(self.Gp, gp)
        for inp, gx in zip(self.Inputs, gxs):
            inp.backward(gx, t)
    
    def grads(self):
        return self.Gp

    def train(self, trainer, regularizer, t):
        if t != self.TrainT:
            self.TrainT = t
            grads = np.mean(self.Gp, axis=0)    # average over batch
            self.TrainerData = trainer.apply(self.Params, grads, self.TrainerData)
            if regularizer is not None:
                regularized = regularizer.apply(self.Regularizable)
            for inp in self.Inputs:
                inp.train(trainer, t)
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
        raise NotImplementedError
        
    def BP(self, gy):
        raise NotImplementedError
    