import keras
from keras.layers import Input, Flatten,Dense, Conv2D, ZeroPadding2D
from keras.layers import Activation, MaxPooling2D, Lambda
from keras.optimizers import RMSprop, SGD, Adam
from keras.models import Model
import keras.backend as K
from keras.datasets import mnist
from keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from utils import Dataset, Dataprep
from time import time

class Siamese1:
    """Implementation of a Siamese network that uses contrastive loss, offering
    usage options for standard convolutional and deep learning models.
    """

    def Dw(self, encods):
        """Calculates the euclidean distance between encodings of images

        Parameters
        ----------
        encods : two tensors

        Description:
                Has the encodings for both images
        Returns
        type: tensor
            Returns the euclidean distance of the two encodings
        """

        x, y = encods
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def Dw_output_shape(self,shapes):
        """Gets the shapes of the encondings that goes as a parameter to the
        Lambda layer that activates the calculation of the euclidean distances

        Parameters
        ----------
        shapes : int
            Shapes of encodings

        Returns
        -------
        type: tuple
            a tuple with the shapes of the encodings

        """
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def digits_inds(self, labels, n_classes=2):
        """Gets the indices for number of different classes we want to train our
        model on

        Parameters
        ----------
        labels : int (0,1)
            Labels of training or testing output
        n_classes : int
            Number of classes we want to train our model with, max value is the
            max number of classes in our data

        Returns
        -------
        type: list
            list of arrays with the indices of different classes in our data, so
            that we can form the pairs
        """
        digit_indices = [np.where(labels==i)[0] for i in range(n_classes)]
        return digit_indices



    def pairs(self,x,y, n_classes=2):
        """Create the genuine and impostor pairs, as in Le Cun's paper:
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Parameters
        ----------
        x : numpy array
            Training or testing data
        y : numpy array
            Output labels of train or test
        n_classes : int
            Number of classes we want to train our model on

        Returns
        -------
        type: Numpy arrays
            Returns numpy arrays with lists of pairs for trianing or testing,
            as well as the respective labels for those pairs so we can train
            the contrastive model
        """
        pairs = []
        labels = []
        digit_indices = self.digits_inds(y, n_classes=n_classes)
        n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1
        for d in range(n_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, n_classes)
                dn = (d + inc) % n_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [0, 1]
        return np.array(pairs), np.array(labels)



    def get_data_prep(self, name = "mnist"):
        """Loads and preprocess the traning and testing data. Replaces the
        standard ways of loading conventional datasets, makes it easier to
        use different ones to experiemnt with the model

        Parameters
        ----------
        name : string
            A string with the name of standard datasets whithin: mnist,
            fashion_mnist and cifar10.

        Returns
        -------
        type:numpy arrays
            Returns numpy arryas of train and test ready to be fed into the
            model.

        """

        dataset = Dataset(name)
        dataprep = Dataprep(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test)
        x_train = dataprep.x_train
        y_train = dataprep.y_train
        x_test = dataprep.x_test
        y_test = dataprep.y_test
        return x_train, y_train, x_test, y_test

    #in the future i intend to include an option for controling some parameters
    #in the model
    def model_1_CNN(self,input_shape):
        """Creates a convolutional network whose last layer is a fully conected
        without activation so that an encoding can be calculated

        Parameters
        ----------
        input_shape : tuple
            Gets the shape of the current dataset being used

        Returns
        -------
        type: keras.models.Model object
            Returns a convolutional model without the last layer activation

        """
        input = Input(shape = input_shape)
        X = ZeroPadding2D((3, 3))(input)
        X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0', activation="relu")(X)
        X = MaxPooling2D((2, 2), name='max_pool')(X)
        X = Flatten()(X)
        X = Dense(10)(X)

        return Model(input,X)

    def model_2_deep_Net(self, input_shape):
        """Creates a neural network with one hidden layer and 20 nodes as in:
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Parameters
        ----------
        input_shape : tuple
            Input shape

        Returns
        -------
        type: keras.models.Model
            Model of a neural network without last layer activation

        """
        input = Input(shape = input_shape)
        X = Flatten()(input)
        X = Dense(20, activation='relu')(X)
        X = Dense(3)(X)

        return Model(input, X)

    def c_loss_1(self, Y, Dw):
        """Contrastive loss function from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Parameters
        ----------
        Y : tensor
            Output labels

        Dw : tensor
            Euclidean distance measure between current pairs.

        Returns
        -------
        type: tensor
            Calculation of contrastive loss given labels, a margin and the
            euclidean distance measure of the encodings
        """
        margin = 1
        loss_gen = (1/2) * K.square(Dw)
        loss_imp = (1/2) *  K.square(K.maximum(margin - Dw, 0))

        c_loss = K.mean((1-Y) * loss_gen + (Y) * loss_imp)

        return c_loss

    def get_model(self,input_shape, nnet = False):
        """Gets the model desired between teo choices: basic convolutional model
        or a neural network model

        Parameters
        ----------
        input_shape : tuple
            Input shape
        nnet : boolean
            If set to true the model chosen is the neural net, otherwise is the
            convolutional network

        Returns
        -------
        type: keras.models.Model object
            Returns the full siamese model that can be compiled.

        """
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        if nnet:
            deep_model = self.model_2_deep_Net(input_shape)
            encoded_a = deep_model(input_a)
            encoded_b = deep_model(input_b)
        else:
            conv_model = self.model_1_CNN(input_shape)
            encoded_a = conv_model(input_a)
            encoded_b = conv_model(input_b)

        d_w = Lambda(self.Dw,output_shape=self.Dw_output_shape)([encoded_a, encoded_b])

        model = Model([input_a, input_b], d_w)
        return model

    def accuracy(self,label, prediction):
        """Compute classification accuracy with distance threshold of 0.5.

        Parameters
        ----------
        label : numpy array
            Labels on training or testing set
        prediction : Numpy array
            Model distance predictions

        Returns
        -------
        type
            Description of returned object.

        """
        acc = K.mean(K.equal(label, K.cast(prediction > 0.5, label.dtype)))

        return acc

    def optimizers(self, name = "adam"):
        """Gives the possibility of choosing standard optmizers

        Parameters
        ----------
        name : string
            Choice of three standard optimizers for siamese: adam, rmsprop and
            stochastica gradient descent.

        Returns
        -------
        type: keras.optimizers object
            Returns type of optimizer
        """
        sgd = keras.optimizers.SGD()
        rms = keras.optimizers.RMSprop()
        adam = keras.optimizers.Adam()
        if name == "adam":
            return adam
        elif name == "rms":
            return rms
        elif name == "sgd":
            return sgd

    def plot_training(self, history, savefig = False):
        """Plots the training accuracy and loss"""

        figure = plt.figure(2)
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(211)
        plt.plot(history.history["accuracy"], c="b")
        plt.title("Accuracy")
        plt.subplot(212)
        plt.plot(history.history["loss"], c="r")
        plt.title("Loss")
        if savefig:
            plt.savefig("training.png")
        plt.show()
        acc_list = history.history["accuracy"]
        print(np.mean(acc_list))






def run():
    siam = Siamese1()
    x_train, y_train, x_test, y_test = siam.get_data_prep(name="fashion_mnist")
    tr_pairs, tr_y = siam.pairs(x_train,y_train)
    te_pairs, te_y = siam.pairs(x_test, y_test)
    input_shape=(x_train.shape[1:])
    siamese_model = siam.get_model(input_shape=input_shape, nnet=True)
    adam = siam.optimizers(name = "sgd")
    siamese_model.compile(optimizer=adam, loss=siam.c_loss_1, metrics = [siam.accuracy])
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(time()), write_graph=True)
    history = siamese_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, validation_split=0.2,
                  batch_size=128, callbacks = [tensorboard], epochs=10)
    siam.plot_training(history)


if __name__ == "__main__":
    run()
