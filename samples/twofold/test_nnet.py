from nnet import Model
from nnet.core_layers import Linear, Input, Flatten
from nnet.activations import Tanh, Sigmoid, SoftPlus
from nnet.losses import CrossEntoryLoss, L2Loss
from nnet.callbacks import Callback, Callbacks
from nnet.trainers import SGD
import numpy as np
import random, math

from generator import generate

nx = 2
ny = 2
nb = 123
w = 0.1


#model = Sequential()
#model.add(Dense(3, input_shape=(nx,), activation="tanh"))
#model.add(Dense(ny, activation="softmax"))


inp = Input((nx,), "input")
l1 = Linear(3, name="l1")(inp)
act = Tanh()(l1)
l2 = Linear(ny, name="l2")(act)
out = CrossEntoryLoss(ny)(l2)
model = Model(inp, out)

sgd = SGD(0.1, momentum = 0.95)

model.compile(trainer=sgd)

import pprint

cfg = model.config()
pprint.pprint(cfg)

m1 = Model.from_config(cfg)
pprint.pprint(m1.config())



pprint.pprint(model.get_params())

model.set_params(model.get_params())

pprint.pprint(model.get_params())


#for t in range(0):
#    x, y_ = generate(nb)
#    losses, metrics = model.fit_batch([x], [y_])
#    #print model.layers()[1].Name, model.layers()[1].Params
#    print losses, metrics
#    #y = model.predict([x])[0]
    
