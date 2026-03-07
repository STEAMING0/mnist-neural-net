import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Initializes the dataset and splits it into training and development sets. The pixel values are normalized to be between 0 and 1 by dividing by 255.
data = pd.read_csv('./data/MNIST.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev/255

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train/255

# Initializes parameters with normal Xavier distribution.
def init_params():
    w1 = np.random.randn(128, 784) * np.sqrt(2 / (784 + 128))
    w2 = np.random.randn(64, 128) * np.sqrt(2 / (128 + 64))
    w3 = np.random.randn(10, 64) * np.sqrt(2 / (64 + 10))
    b1 = np.zeros((128, 1))
    b2 = np.zeros((64, 1))
    b3 = np.zeros((10, 1))

    return w1, b1, w2, b2, w3, b3
# Neural network functions for the 2 hidden layers: tanh activation function, softmax for output layer, one-hot encoding for labels, forward propagation, backward propagation, parameter updates, prediction generation, and accuracy calculation.
def tanh(z):
    return np.tanh(z)

def dtanh(z):
    return 1 - np.tanh(z) ** 2

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    expz = np.exp(z_shifted)
    return expz / np.sum(expz, axis=0, keepdims=True)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def forward_prop(w1, b1, w2, b2, w3, b3, X):
    z1 = np.dot(w1, X) + b1
    a1 = tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = tanh(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)

    return z1, a1, z2, a2, z3, a3

def backward_prop(z1, a1, z2, a2, a3, w2, w3, X, Y):

    m = Y.size
    one_hot_Y = one_hot(Y)

    dz3 = a3 - one_hot_Y
    dw3 = (1/m) * np.dot(dz3, a2.T)
    db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)

    dz2 = np.dot(w3.T, dz3) * dtanh(z2)
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * dtanh(z1)
    dw1 = (1/m) * np.dot(dz1, X.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2, dw3, db3

def update_params(w1, b1, w2, b2, w3, b3,
                  dW1, db1, dW2, db2, dW3, db3, alpha):
    w1 -= alpha * dW1
    b1 -= alpha * db1
    w2 -= alpha * dW2
    b2 -= alpha * db2
    w3 -= alpha * dW3
    b3 -= alpha * db3

    return w1, b1, w2, b2, w3, b3

def get_predictions(a3):
    return np.argmax(a3, axis=0)

def get_accuracy(p, Y):
    return np.mean(p == Y)

def gradient_descent(X, Y, alpha, epochs):
    w1, b1, w2, b2, w3, b3 = init_params()
    for i in range(epochs):
        z1, a1, z2, a2, z3, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X)

        dW1, db1, dW2, db2, dW3, db3 = backward_prop(
            z1, a1, z2, a2, a3, w2, w3, X, Y
        )
        w1, b1, w2, b2, w3, b3 = update_params(
            w1, b1, w2, b2, w3, b3,
            dW1, db1, dW2, db2, dW3, db3,
            alpha
        )
        if i % 100 == 0:
            preds = get_predictions(a3)
            a = get_accuracy(preds, Y)
            print(f"Epoch {i} Accuracy: {a*100:.2f}%")

    return w1, b1, w2, b2, w3, b3

def make_prediction(X, w1, b1, w2, b2, w3, b3):
    _, _, _, _, _, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
    return np.argmax(a3, axis=0)

