# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os


if __name__ == '__main__':
    # Load the data from Keras.datasets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # let's print the shape before we reshape and normalize
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)
    # building the input vector from the 28x28 pixels
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255
    # print the final input shape ready for training
    print("Train matrix shape", X_train.shape)
    print("Test matrix shape", X_test.shape)
    print(np.unique(y_train, return_counts=True))

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)

    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))                            
    model.add(Dropout(0.20)) # To prevent overfitting
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.20)) 
    model.add(Dense(n_classes, activation='softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model and saving metrics in history
    history = model.fit(X_train, Y_train,
            batch_size=128, epochs=20,
            verbose=2,
            validation_data=(X_test, Y_test))
    
    # plotting the metrics
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim([2, 20])
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim([2, 20])
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    #plt.savefig("CNN Valiadation Accuracy")
    plt.show()
    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
    print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])