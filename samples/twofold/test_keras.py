import numpy as np

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD

from generator import generate

nx = 2
ny = 2
nb = 123
w = 0.1

model = Sequential()
model.add(Dense(3, input_shape=(nx,), activation="tanh", name="dense"))   #, kernel_initializer='random_uniform'))
model.add(Dense(ny, activation="softmax", name="out"))                  #, kernel_initializer='random_uniform'))

sgd = SGD(lr=0.1, decay=0.0, momentum=0.95, nesterov=False)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

print model.layers[0].get_weights()


for t in range(100):
    x, y_ = generate(nb)
    print model.train_on_batch([x], [y_])
    #print model.layers[0].name, model.layers[0].get_weights()
    