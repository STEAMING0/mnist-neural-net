# MNIST Neural Network From Scratch

This project trains a feedforward neural network to classify digits `0-9` from MNIST image data. It is intended as a learning-focused implementation of core neural network mechanics:

- parameter initialization
- forward propagation
- backpropagation
- softmax classification
- one-hot label encoding
- gradient descent optimization

## Model Architecture

The network uses:

- input layer: `784` features (`28x28` grayscale image flattened)
- hidden layer 1: `128` neurons with `tanh`
- hidden layer 2: `64` neurons with `tanh`
- output layer: `10` neurons with `softmax`

Weights are initialized with Xavier-style scaling, and pixel values are normalized to the `0-1` range before training.

## Features

- neural network implemented from scratch in Python
- two hidden layers with `tanh` activations
- softmax output for multiclass digit prediction
- manual forward and backward propagation
- gradient descent training loop with accuracy logging
- interactive prediction display for training examples
- digit visualization with `matplotlib`


