# Simple-DeepLearning-System

![mit license](https://img.shields.io/github/license/Att100/Simple-DeepLearning-System)

## Introduction
This is a simple deep learning system with numpy. It include a `autograd` system which is the basis of the whole system while all network layers and tensor operations that implemented are based on the simple dynamic computation graph. We now support fundamental mathmatics operation and several layers, including `Linear`,  `ReLU`, `Dropout1d`, `BatchNorm1d`, `Softmax`, and some pre-defined loss functions such as `MSELoss` and `CrossEntropyLoss`. We plan to support some core components of convolution nerual networks and CUDA in the future. 


## Quick Start

- Download MNIST dataset and place it in ./dataset/mnist, 

    + http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    + http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    + http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    + http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

    then it should satisfy the structure below

    ```
    ├─dataset
        └─mnist
            ├─ source.txt
            ├─ t10k-images.idx3-ubyte
            ├─ t10k-labels.idx1-ubyte
            ├─ train-images.idx3-ubyte
            └─ train-labels.idx1-ubyte
    ```

- Run example of linear regression

  ```
  python ex_linear_regression.py
  ```

- Run example of MLP on MNIST dataset

  ```
  python ex_mlp_mnist.py
  ```