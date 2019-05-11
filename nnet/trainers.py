class Trainer:
    
    def apply(self, params, grads, data):
        return params, data
    
class SGD(Trainer):
    
    def __init__(self, eta):
        self.Eta = eta
        
    def apply(self, params, grads, train_data):
        for p, g in zip(params, grads):
            p -= g*self.Eta
        return train_data
            
            