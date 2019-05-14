from nnet import Model
from nnet.core_layers import Linear, Input, Flatten
from nnet.activations import Tanh, Sigmoid, SoftPlus
from nnet.losses import CrossEntropyLoss, L2Loss
from nnet.callbacks import Callback, Callbacks
import numpy as np
import random
from nnet.trainers import SGD

inp = Input((28, 28), "images")
top = Flatten()(inp)
top = Tanh()(Linear(28*28*10, name="l1")(top))
top = Tanh()(Linear(1000, name="l2")(top))
top = Linear(10, name="out")(top)
loss = CrossEntropyLoss("LogLoss")(top)

model = Model(inp, loss)

cfg = model.config()

m1 = Model.from_config(cfg)

