class Callback(object):
    
    #def onBatchEnd(self, samples, total_samples, losses, metrics):
    #    pass
        
    #def onEpochEnd(self, epoch, samples, total_samples, losses, metrics):
    #    pass

    pass
    
class Callbacks(object):
    
    def __init__(self, callbacks = []):
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.Lst = callbacks
        
    def onBatchEnd(self, *params, **args):
        for c in self.Lst:
            if hasattr(c, "onBatchEnd"):
                c.onBatchEnd(*params, **args)
                
    def onEpochEnd(self, *params, **args):
        for c in self.Lst:
            if hasattr(c, "onEpochEnd"):
                c.onEpochEnd(*params, **args)
