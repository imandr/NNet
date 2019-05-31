from nnet import Model
from nnet.core_layers import Linear, Input, Flatten
from nnet.activations import Tanh, Sigmoid, SoftPlus
from nnet.losses import CrossEntropyLoss, L2Loss
from nnet.callbacks import Callback, Callbacks
import numpy as np
import random
from nnet.trainers import SGD
from progress.bar import Bar

mnist_data = np.load("mnist.npz")
train_images, train_labels = mnist_data["train_images"], mnist_data["train_labels"]
test_images, test_labels = mnist_data["test_images"], mnist_data["test_labels"]

print "Loaded %d MNIST images: %s %s" % (len(train_images), train_images.shape, train_labels.shape)

inp = Input((28, 28), "images")
top = Flatten()(inp)
top = Tanh()(Linear(28*28*10, name="l1")(top))
top = Tanh()(Linear(1000, name="l2")(top))
top = Linear(10, name="out")(top)
loss = CrossEntropyLoss("LogLoss")(top)

model = Model(inp, loss)
model.compile(trainer=SGD(0.01, momentum=0.9))

def calc_grads(model, mbsize, x, y_):
    
    assert len(x) == len(y_)
    
    sumgrads = [np.zeros_like(p) for p in model.get_params()]
    sumloss = 0.0
    N = len(x)
    bar = Bar("Calculating gradients...", suffix="%(index)d/%(max)d - %(percent)d%% - loss:%(loss)f - acc:%(acc).1f%%", max=N)

    for i in range(0, len(x), mbsize):        
        n = len(x[i:i+mbsize])
        y, losses, metrics = model.forward([x[i:i+mbsize]], [y_[i:i+mbsize]])
        sumloss += losses[0]*n
        model.backward([y_[i:i+mbsize]])
        grads = model.get_grads()
        for s, g in zip(sumgrads, grads):
            s += g * n
        bar.loss = losses[0]
        bar.acc = metrics[0]*100.0
        bar.next(n)
    bar.loss = sumloss/N
    bar.finish()
    return sumloss/N, [g/N for g in sumgrads] 

def calc_deltas(grads, eta, momentum, last_deltas):
    if last_deltas is None:
        last_deltas = [np.zeros_like(g) for g in grads]
    
    return [d*momentum - eta*g for d, g in zip(last_deltas, grads)]
        
def apply_deltas(model, deltas):
    params = [d + p for d, p in zip(deltas, model.get_params())]
    model.set_params(params)

deltas = None
    
for _ in range(10):
    loss, grads = calc_grads(model, 500, train_images, train_labels)
    print loss
    deltas = calc_deltas(grads, 0.1, 0.9, deltas)
    apply_deltas(model, deltas)
    
    
    

 