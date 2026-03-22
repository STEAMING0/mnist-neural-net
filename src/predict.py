import numpy as np
from train import x_train, y_train, make_prediction, gradient_descent
from utils import display_prediction

if __name__ == "__main__":
    w1, b1, w2, b2, w3, b3 = gradient_descent(x_train, y_train, alpha=0.1, epochs=500)
x
    while True:
        user = input("Enter an index of the training set to see prediction (or 'q' to quit): ")

        if user.lower() == 'q':
            break

        if not user.isdigit() or int(user) < 0 or int(user) >= x_train.shape[1]:
            print(f"Please enter a valid index between 0 and {x_train.shape[1]-1}")
            continue

        display_prediction(
            int(user),
            x_train,
            y_train,
            make_prediction,
            w1, b1, w2, b2, w3, b3
        )
