from .core_layers import Input
from .loss import LossLayer
from .trainers import SGD
import numpy as np

def Model(object):
    
    def __init__(self, inputs, outputs, losses):
        self.Inputs = inputs if isinstance(inputs, list) else [inputs]
        for inp in self.Inputs:
            assert isinstance(inp, Input)
        self.Outputs = outputs if isinstance(outputs, list) else [outputs]
        self.T = 0
        
    def compile(self, losses, trainer=None, regularizer=None):
        losses = losses if isinstance(losses, list) else [losses]
        assert len(losses) == len(self.Outputs)
        for loss, out in zip(losses, self.Outputs):
            assert isinstance(loss, LossLayer)
            loss.link(out)
        self.Trainer = trainer or SGD(0.001)
        self.Regularizer= regularizer
        
    def resetState(self):
        for out in self.Outputs:
            out.reset_state(self.T)
        
    def predict(self, inputs, next_t = True, reset_state = True):
        assert len(inputs) == len(self.Inputs)
        if next_t:  
            self.T += 1
            if reset_state:
                self.resetState()
        for inp, x in zip(self.Inputs, inputs):
            inp.set(x, self.T)
        return [out.forward(self.T) for out in self.Outputs]

    def fit_batch(self, xs, ys_):
        self.T += 1
        self.resetState()
        assert len(xs) == len(self.Inputs)
        ys = self.predict(xs)
        losses_before = [loss.loss(yi_, self.T) for loss, yi_ in zip(self.Losses, ys_)]
        for loss, yi_ in zip(self.Losses, ys_):
            loss.backward(yi_, self.T)
        for out in self.Outputs:
            out.train(self.Trainer, self.Regularizer, self.T)
        return self.evaluate(xs, ys_)
            
    def evaluate(self, xs, ys_):
        self.T += 1
        self.resetState()
        losses = [loss.loss(yi_, self.T) for loss, yi_ in zip(self.Losses, ys_)]
        metrics = [loss.Metric(out.Y, ys_) for (out, loss) in zip(self.Outputs, self.Losses)]
        return losses, metric
        
    def fit_dataset(self, xs, ys, batch_size, epochs = 1, shuffle = True, callbacks = None):
        assert len(xs) == len(ys)
        N = len(xs)
        if shuffle:
            inx = np.arange(len(N))
        xx, yy = xs, ys
        samples = 0
        total_samples = N*epochs
        losses, metrics = None, None
        for epoch in range(epochs):

            if shuffle:
                np.shuffle(inx)
                xx, yy = xs[inx], ys[inx]
            
            for i in range(0, N, batch_size):
                last_x, last_y = xx[i:i+batch_size], yy[i:i+batch_size
                losses, metrics = self.fit_batch(xx[i:i+batch_size], yy[i:i+batch_size])
                samples += len(xx[i:i+batch_size])
                
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
        
    