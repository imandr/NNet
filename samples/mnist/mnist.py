from nnet import Model
from nnet.core_layers import Linear, Input, Flatten
from nnet.activations import Tanh, Sigmoid, SoftPlus
from nnet.losses import CrossEntropyLoss, L2Loss
from nnet.callbacks import Callback, Callbacks
import numpy as np
import random
from nnet.trainers import SGD

mnist_data = np.load("mnist.npz")
train_images, train_labels = mnist_data["train_images"], mnist_data["train_labels"]
test_images, test_labels = mnist_data["test_images"], mnist_data["test_labels"]

print "Loaded %d mnis images: %s %s" % (len(train_images), train_images.shape, train_labels.shape)

inp = Input((28, 28), "images")
top = Flatten()(inp)
top = Tanh()(Linear(28*28*10, name="l1")(top))
top = Tanh()(Linear(1000, name="l2")(top))
top = Linear(10, name="out")(top)
loss = CrossEntropyLoss("LogLoss")(top)

model = Model(inp, loss)
model.compile(trainer=SGD(0.01, momentum=0.9))

def print_image(img):
    for row in img:
        line = ''.join([
            '#' if x > 0.7 else (
                '+' if x > 0.3 else (
                    '.' if x > 0 else' ')) for x in row])
        print line
        

        

class cb(Callback):
    
    def onEpochEnd(self, epoch, samples, total_samples, losses, metrics):
        print "epoch end:", epoch, samples, total_samples, losses, metrics
        
        
        

    def onBatchEnd(self, samples, total_samples, losses, metrics):
        print "batch end:", samples, total_samples, losses, metrics
        i = random.randint(0, len(test_images)-1)
        img, label_ = test_images[i], np.argmax(test_labels[i])
        print_image(img)
        probs = model.predict([img.reshape((1,)+img.shape)])
        label = np.argmax(probs)
        print label_, label, probs

callbacks = Callbacks(cb())

model.fit_dataset([train_images], [train_labels], 100, shuffle=True, callbacks = callbacks)

