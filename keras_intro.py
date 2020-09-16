#!/usr/bin/python3
"""
first tensorflow projectt
@Author: Shaan Gagneja
@Email: sgagneja@wisc.edu
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from tensorflow_core.python.keras.layers import Activation

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_dataset(training=True):
    """
    takes an optional boolean argument and returns the dataset
    :param training: whether training is occuring or not
    :return: the dataset of images and labels
    """
    fashion_mnist = keras.datasets.fashion_mnist

    # gets labels and images from load data
    (train_images, train_labels), (test_images, test_labels) = np.array(fashion_mnist.load_data())

    # determines if test or training data should be returned
    if training == False:
        return test_images, test_labels
    return train_images, train_labels


def print_stats(images, labels):
    """
    takes the dataset and labels produced by the previous function and prints several statistics about the data
    :param images: dataset of images
    :param labels: dataset of labels
    """
    print(len(images))
    print(len(images[0]), "x", len(images[0]))
    # prints number of images for each label
    for i in range(len(class_names)):
        x = 0
        for j in labels:
            if i == j:
                x += 1
        print("%d. %s - %d" % (i, class_names[i], x))


def view_image(image, label):
    """
    takes a single image as an array of pixels and displays an image
    :param image: image to be displayed
    :param label: label of image
    """
    # reshapes image
    image = np.reshape(image, (28, 28))

    # creates the figure for image to go into
    fig, im = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # sets up the figure correctly
    im.set_title(label)
    i = im.imshow(image, aspect='equal')
    fig.colorbar(i, ax=im)

    # shows the image
    matplotlib.pyplot.show()


def build_model():
    """
    :return: an untrained neural network as specified below
    """
    # creates sequential model with flattened, two dense layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10)
    ])
    # compiles the model
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def train_model(model, images, labels, T):
    """
    takes the model produced by the previous function and the images and labels produced by the first function
    and trains the data for T epochs
    :param model: model produced by previous function
    :param images: set of images
    :param labels: set of labels for images
    :param T: the number of epochs
    """
    model.fit(images, labels, epochs=T)


def evaluate_model(model, images, labels, show_loss=True):
    """
    takes the trained model produced by the previous function and the test image/labels, and prints the evaluation
    statistics, displaying the loss metric value if and only if the optional parameter has not been set to False
    :param model: trained model
    :param images: set of images
    :param labels: set of labels for images
    :param show_loss: whether loss metric should be shown
    """
    test_loss, test_accuracy = model.evaluate(images, labels, verbose=0)
    if show_loss == False:
        print("Accuracy: %12.2f" % (test_accuracy * 100), "%")
    else:
        print("Loss: %12.2f" % test_loss)
        print("Accuracy: %12.2f" % (test_accuracy * 100), "%")


def predict_label(model, images, index):
    """
    takes the trained model and test images, and prints the top 3 most likely labels for the image at the
    given index, along with their probabilities
    :param model: trained model
    :param images: set of images
    :param index: set of labels for images
    """
    model.add(Activation('softmax'))
    ind = model.predict(images)[index]
    # list of indices of top 3 likely labels for image at given index
    top_3 = reversed(sorted(range(len(ind)), key=lambda i: ind[i])[-3:])
    for i in top_3:
        print("%s: %12.2f" % (class_names[i], ind[i] * 100), "%")
