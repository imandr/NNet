class Trainer:
    
    def apply(self, params, grads, data):
        return params, data
    
class SGD(Trainer):
    
    def __init__(self, eta, momentum = 0.0):
        self.Eta = eta
        self.Momentum = momentum
        
    def apply(self, params, grads, train_data):
        if train_data is not None:
            dp = [m*self.Momentum - g*self.Eta for m, g in zip(train_data, grads)]
        else:
            dp = [                - g*self.Eta for g in grads]
        
        for p, delta in zip(params, dp):
            p += delta
        return dp if self.Momentum != 0.0 else None
            
            