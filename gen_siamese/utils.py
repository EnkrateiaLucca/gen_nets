
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import keras
import importlib as imp
import numpy as np
from keras.datasets import mnist

class Dataset:
    """Gives standard datasets for supervised machine learning tasks by using
    keras.datasets. Gives the possibility to preprocess and plot the images.

    Parameters
    ----------
    name : str
        A standard dataset among
        ["mnist", "fashion_mnist", "cifar10", "cifar100"].
    label_mode : str
        Specific attribute for CIFAR100 labelling taken directly from keras.
    """
    def __init__(self, name = None, label_mode = 'fine',
                 num_samples_from_training= None, num_classes = None):
        self.name = name
        if self.name is 'cifar100':
            (self.x_train, self.y_train),\
            (self.x_test, self.y_test) = self.data_set.load_data(label_mode=label_mode)
        else:
            (self.x_train, self.y_train),\
            (self.x_test, self.y_test) = self.data_set.load_data()
        self.classes = np.unique(self.y_train)
        if num_classes is not None:
            self.limit_to_n_classes(num_classes)
        if num_samples_from_training is not None:
            perm = np.random.permutation(len(self.x_train))
            self.x_train = self.x_train[perm][:num_samples_from_training]
            self.y_train = self.y_train[perm][:num_samples_from_training]
        #self.y_train = keras.utils.np_utils.to_categorical(self.y_train)
        self.load_data = (self.x_train, self.y_train,self.x_test, self.y_test)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        AVAILABLE_DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
        if new_name in AVAILABLE_DATASETS:
            self._name = new_name
            self.data_set = imp.import_module("keras.datasets.{}".format(new_name))
        else:
            raise ValueError("Please select one of" +
                             " the dataset available" +
                             " in {}".format(AVAILABLE_DATASETS))

class Dataprep:
    """Preprocesses the data: convertion to float, normalization,
    reshaping and standardization"""
    def __init__(self,x_train,y_train,x_test,y_test):
        """"""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.convert_float()
        self.normalize()
        self.reshape()

    def convert_float(self):
        """Converts inputs values to floats"""
        self.x_train = self.x_train.astype("float32")
        self.x_test = self.x_test.astype("float32")

    def reshape(self):
        """Reshapes the input """
        if self.x_train.ndim < 4:
            self.x_train = self.x_train.reshape(self.x_train.shape[0],self.x_train.shape[1], self.x_train.shape[2],1)

            self.x_test = self.x_test.reshape(self.x_test.shape[0],self.x_test.shape[1], self.x_test.shape[2],1)

    def normalize(self):
        """Divides the input by the number of pixels to avoid values that
        are too different"""
        self.x_train /= 255.0
        self.x_test /= 255.0

    def standardize(self):
        """"""
        pass

# if __name__ == "__main__":
#     dset = Dataset(name = "cifar10")
#     x_train, y_train, x_test, y_test = dset.x_train, dset.y_train, dset.x_test, dset.y_test
#     print(x_train.shape[])
