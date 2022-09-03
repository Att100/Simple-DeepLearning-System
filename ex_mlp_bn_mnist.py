import numpy as np
import struct
import os

import TinyDL
import TinyDL.nn as nn
import TinyDL.nn.functional as F
from TinyDL import Tensor
import TinyDL.optimizer as optim


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)
    lab = labels.tolist()
    l = []
    for i in range(len(lab)):
        label = [0 for j in range(10)]
        label[lab[i]] = 1
        l.append(label)
    labels = np.array(l)
    return images, labels


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear1 = nn.Linear(784, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.linear3 = nn.Linear(16, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.linear1(x)))
        out = F.relu(self.bn2(self.linear2(out)))
        out = self.linear3(out)
        return out


if __name__ == "__main__":
    path = "D:\\Workspace\\project\\TinyDLProject\\dataset\\mnist"
    batchsize = 600
    te_batchsize = 100
    epochs = 50

    model = Net()
    model.init()
    citeration = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate=1e-3)

    inputs, targets = load_mnist(path, 'train')
    te_inputs, te_targets = load_mnist(path, 't10k')
    inputs = inputs / 255  # normalize
    te_inputs = te_inputs / 255  # normalize

    n_tr, n_te = inputs.shape[0], te_inputs.shape[0]
    steps_tr, steps_te = int(n_tr/batchsize), int(n_te/te_batchsize)

    model.train()
    for i in range(epochs):
        loss_ = 0.0
        acc_ = 0.0
        for j in range(steps_tr):
            x = Tensor(
                data=inputs[j*batchsize:(j+1)*batchsize, :],
                requires_grad=True,
                name = "x")
            target = Tensor(data=targets[j*batchsize:(j+1)*batchsize, :], name='y')

            optimizer.zero_grad()
            out = model(x)
            loss = citeration(out, target)

            loss_ += loss.data
            acc_ += np.sum(np.argmax(out.data, axis=1)==np.argmax(target.data, axis=1)) / batchsize

            loss.backward()
            optimizer.step()
        print("epoch: {}, loss: {:.5f}, acc: {:.5f}".format(i+1, loss_ / steps_tr, acc_ / steps_tr))

    te_acc_ = 0
    model.eval()
    for j in range(steps_te):
        x = Tensor(
            data=te_inputs[j*batchsize:(j+1)*batchsize, :],
            requires_grad=True,
            name = "x")
        out = model(x)
        target = Tensor(data=te_targets[j*batchsize:(j+1)*batchsize, :], name='y')
        te_acc_ += np.sum(np.argmax(out.data, axis=1)==np.argmax(target.data, axis=1)) / te_batchsize
    print("test_acc: {:.5f}".format(te_acc_ / steps_te))