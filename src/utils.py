import numpy as np
import matplotlib.pyplot as plt

def display_prediction(index, x_train, y_train, make_prediction,
                       w1, b1, w2, b2, w3, b3):

    image = x_train[:, index:index+1]

    prediction = make_prediction(image, w1, b1, w2, b2, w3, b3)
    y_sample = y_train[index]

    print("Prediction:", prediction[0])
    print("Label:", y_sample)

    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()