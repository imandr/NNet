from .core_layers import Input
from .losses import LossLayer
from .trainers import SGD
from .layer import Layer
import numpy as np

class Model(object):
    
    def __init__(self, inputs, outputs):
        self.Inputs = inputs if isinstance(inputs, list) else [inputs]
        for inp in self.Inputs:
            assert isinstance(inp, Input)
        self.Outputs = outputs if isinstance(outputs, list) else [outputs]
        self.T = 0
    
    def layers_rec(self, top, dct, out):
        for inp in top.Inputs:
            self.layers_rec(inp, dct, out)
        lid = top.ID
        if not lid in dct:
            dct[lid] = top
            out.append(top)

    @property
    def layers(self):
        dct = {}
        out = []
        for top in self.Outputs:
            self.layers_rec(top, dct, out)
        return out
        
    def compile(self, trainer=None, regularizer=None):
        for out in self.Outputs:
            assert isinstance(out, LossLayer), "Output layers of a trainable model must be LossLayers"
        self.Trainer = trainer or SGD(0.01)
        self.Regularizer= regularizer
        
    def resetState(self):
        # will not increment T !
        for out in self.Outputs:
            out.resetState(self.T)
            
    def tick(self):
        self.T += 1
        
    def predict(self, xs):
        assert len(xs) == len(self.Inputs)
        self.tick()
        for inp, x in zip(self.Inputs, xs):
            inp.set(x, self.T)
        return [out.forward(self.T) for out in self.Outputs]

    def forward(self, xs, ys_=None):
        y = self.predict(xs)
        losses, metrics = self.evaluate(ys_)
        return y, losses, metrics
        
    run = forward
        
    def backward(self, ys_):
        for out, yi_ in zip(self.Outputs, ys_):
            out.backward(yi_, self.T)
            
    def train(self):
        for out in self.Outputs:
            out.train(self.Trainer, self.Regularizer, self.T)

    def evaluate(self, ys_):
        if ys_ is None:
            return None, None
        losses = [loss.loss(yi_) for loss, yi_ in zip(self.Outputs, ys_)]
        metrics = [loss.Metric(out.Y, yi_) for (out, yi_) in zip(self.Outputs, ys_)]
        return losses, metrics

    def fit_batch(self, xs, ys_):
        self.resetState()
        self.forward(xs, ys_)
        self.backward(ys_)
        self.train()
        self.predict(xs)
        return self.evaluate(ys_)
            
    def fit_dataset(self, xs, ys, batch_size, epochs = 1, shuffle = False, callbacks = None):
        #
        # shuffle is not working yet
        #
        shuffle = False
        N = len(xs[0])
        for x in xs:
            assert len(x) == N
        for y in ys:
            assert len(y) == N
        if shuffle:
            inx = np.arange(N)
        xx, yy = xs, ys
        samples = 0
        total_samples = N*epochs
        losses, metrics = None, None
        for epoch in range(epochs):

            if shuffle:
                np.random.shuffle(inx)
                xx = [x[inx] for x in xs]
                yy = [y[inx] for y in ys]
            
            for i in range(0, N, batch_size):
                losses, metrics = self.fit_batch(
                    [x[i:i+batch_size] for x in xx],
                    [y[i:i+batch_size] for y in yy]
                )
                samples += len(xx[0][i:i+batch_size])
                
                if callbacks:
                    callbacks.onBatchEnd(samples, total_samples, losses, metrics)
            if callbacks:
                callbacks.onEpochEnd(epoch, samples, total_samples, losses, metrics)
            
        return losses, metrics
        
    def fit_generator(self, generator, batch_size, samples_per_epoch, epochs = 1, callbacks = None):
        samples = 0
        total_samples = samples_per_epoch * epochs
        losses, metrics = None, None
        for epoch in range(epochs):

            if reset_state in ("both","epoch"):
                self.resetState()
                
            samples_this_epoch = min(samples_per_epoch, total_samples-samples)
            while samples_this_epoch > 0:
                n = min(batch_size, samples_this_epoch)
                xs, ys = generator(n)
                assert len(xs) <= n and len(xs) > 0
                assert len(ys) == len(xs)
                n = len(xs)
                losses, metrics = self.fit_batch(xs, ys,)
                samples += n
                samples_this_epoch -= n
                if callbacks:
                    callbacks.onBatchEnd(samples, total_samples, losses, metrics)
            if callbacks:
                callbacks.onEpochEnd(epoch, samples, total_samples, losses, metrics)
            
        return losses, metrics
        
    def config(self):
        lst = self.layers
        for i, layer in enumerate(lst):
            layer.ID = i
        out = dict(
            inputs = [inp.ID for inp in self.Inputs],
            layers = [layer.config() for layer in self.layers],
            outputs = [out.ID for out in self.Outputs]
        )
        return out
        
    @staticmethod
    def from_config(cfg):
        layers = {}
        for l in cfg["layers"]:
            layers[l["id"]] = Layer.from_config(l)
        for l in cfg["layers"]:
            layer = layers[l["id"]]
            inputs = [layers[lid] for lid in l.get("inputs", [])]
            if inputs:
                layer.link(inputs)
        inputs = [layers[lid] for lid in cfg["inputs"]]
        outputs = [layers[lid] for lid in cfg["outputs"]]
        return Model(inputs, outputs)
                
        
    def get_params(self):
        out = []
        for layer in self.layers:
            p = layer.get_params()
            if p:   out += p
        return out
        
    def set_params(self, params):
        i = 0
        for layer in self.layers:
            n = layer.nparams()
            if n > 0:
                layer.set_params(params[i:i+n])
                i += n
        
    def get_grads(self):
        out = []
        for layer in self.layers:
            g = layer.get_grads()
            if g:   out += g
        return out
        
        