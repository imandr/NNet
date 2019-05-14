import numpy as np

def read_file(fn):
    labels = []
    images = []
    with open(fn, "r") as f:
        hdr = f.readline()
        for l in f.readlines():
            l = l.strip()
            if l:
                data = l.split(",")
                label = int(data[0])
                image = [float(x)/256.0 for x in data[1:]]
                assert len(image) == 28*28
                labels.append(label)
                images.append(image)
    labels = np.array(labels)
    labels_onehot = np.zeros((len(labels), 10))
    for i in range(10):
        labels_onehot[labels==i,i] = 1.0 
    return labels_onehot, np.array(images).reshape((len(images),28,28))

train_labels, train_images = read_file("mnist_train.csv")
test_labels, test_images = read_file("mnist_test.csv")

np.savez("mnist.npz", 
    train_labels = train_labels, train_images = train_images,
    test_labels = test_labels, test_images = test_images)