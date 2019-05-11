from .layer import Layer

class Linear(Layer):
    
    def __init__(self, nout, name=None):
        Layer.__init__(self, name)
        self.NOut = nout
        
    def init(self, inputs):
        assert len(inputs) == 1
        inp = inputs[0]
        n_in = inp.Shape[-1]
        W_shape = (self.n_in, self.n_out)
        rng = np.random.RandomState()
        self.W = rng.normal(size=W_shape, scale=1.0/math.sqrt(n_in))
        self.b = np.zeros(self.NOut)
        self.Shape = inp.Shape[:-1] + [self.NOut]
        self.Params = (self.W, self.b)
        self.Regularizable = (self.W,)

    def FP(self, xs, state):
        assert len(xs) == 1
        x = xs[0]
        y = np.dot(x[0], self.W) + self.b
        rerurn y, state
        
    def BP(self, gy):
        #print "linear.bprop: x:", x[0].shape, "   gy:", gY.shape
        x = self.X[0]
        n_mb = len(x)
        inp_flat = x.reshape((-1, self.n_in))
        g_flat = gy.reshape((-1, self.n_out))
        gW = np.dot(inp_flat.T, g_flat)/n_mb    # [...,n_in],[...,n_out]
        gb = np.mean(g_flat, axis=0)
        gx = np.dot(gy, self.W.T)
        return [gx], [gW, gb]
        
class Merge(Layer):
    
    def __init__(self, axis=0, name=None):
        Layer.__init__(self, name)
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

class Input(Layer):
    
    def __init__(self, shape):
        self.Shape = shape
        
    def set(self, value, t):
        self.Xs = [value]
        self.newTick(t)
        
    def forward(self, t=None):
        assert t == self.Tick
        self.Y = self.Xs[0]
        return self.Y
        
    def BP(self, gy):
        return (), ()
        
        
        
        