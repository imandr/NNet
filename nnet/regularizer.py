class Regularizer(object):
    
    def __call__(self, params):
        return params
        
class L1Reguralizer(Regularizer):
    
    def __init__(self, decay):
        self.Decay = decay
        
    def __call__(self, params):
        return tuple(p*(1-self.Decay) for p in params)