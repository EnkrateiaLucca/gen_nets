"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import keras
from time import time
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from siamese1 import Siamese1
from utils import Dataset, Dataprep
import numpy as np
import os

def train_and_score(network, dataset, curr_gen_num):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """

    ########################################################
    layers = network['nb_layers']
    act = network['activation']
    optimizer = network['optimizer']

    siam = Siamese1()
    x_train, y_train, x_test, y_test = siam.get_data_prep(name=dataset)
    input_shape=(x_train.shape[1:])

    tr_pairs, tr_y = siam.pairs(x_train,y_train)
    te_pairs, te_y = siam.pairs(x_test, y_test)

    siamese_model = siam.get_model(input_shape=input_shape,nb_layers=layers, activation=act)
    siamese_model.compile(optimizer=optimizer, loss=siam.c_loss_1, metrics = [siam.accuracy])

    history = siamese_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, validation_split=0.2,
                  batch_size=64, epochs=10)
    ind_acc = history.history['accuracy']
    score = np.mean(ind_acc)

    return score, ind_acc
