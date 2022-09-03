import matplotlib.pyplot as plt
import random
import os

import TinyDL
from TinyDL import Tensor


if __name__ == "__main__":
    os.chdir("../../")
    
    k, b = 2, 3
    n_samples = 10
    reduction = 1
    noise_range = [-2, 2]
    epochs = 500
    lr = 0.001

    samples = []
    for i in range(n_samples):
        x = (i+1)*reduction
        y_true = k*(i+1)*reduction + b
        y = y_true + random.uniform(noise_range[0], noise_range[1])
        samples.append([x, y])
    random.shuffle(samples)

    k = Tensor(data=0, requires_grad=True, name="k")
    b = Tensor(data=0, requires_grad=True, name="b")

    for i in range(epochs):
        loss_ = 0
        random.shuffle(samples)
        for j in range(n_samples):
            x = Tensor(data=samples[j][0], requires_grad=True, name="x")
            y = Tensor(data=samples[j][1], requires_grad=True, name="y")
            pred = k * x + b
            loss = TinyDL.pow(pred-y, 2)
            loss_ += loss.data
            
            loss.backward()

            k.data -= k.grad * lr
            b.data -= b.grad * lr

            k.zero_grad()
            b.zero_grad()
  
        print("loss: ", loss_ / n_samples)

    print("k: ", k.data, "b: ", b.data)

    plt.scatter(
        [samples[i][0] for i in range(n_samples)], 
        [samples[i][1] for i in range(n_samples)])
    plt.plot([i*k.data+b.data for i in range(int(n_samples*reduction))], c='r')
    plt.show()
