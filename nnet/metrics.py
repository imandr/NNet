import numpy as np

class Metric(object):
    
    def __init__(self, label = None):
        self.Name = label
        
    def __call__(self, y, y_):
        raise NotImplementedError
        
class MSE(Metric):
    
    def __init__(self, label = "mse"):
        Metric.__init__(self, label)
        
    def __call__(self, y, y_):
        return np.mean(np.sum(np.square(y-y_), axis=-1), axis=0)
        
class ClassificationError(Metric):
    
    def __init__(self, label = "error"):
        Metric.__init__(self, label)
        
    def __call__(self, y_, y):
        return float(np.sum(np.argmax(y_, axis=-1) != np.argmax(y, axis=-1)), axis=0)/len(y)
    
        
