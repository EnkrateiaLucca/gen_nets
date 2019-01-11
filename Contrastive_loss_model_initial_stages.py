from scipy.spatial import distance
from keras.layers import Input, Flatten, Dense

class Siamese1:
    """
    Siamese Network implementing contrastive loss and options for
    different standard models.

    Attributes
    ---------

    load_data : str
        The path to ....


    """
    def __init__(self, load_data = False):
        self.load_data = load_data


    def euclidean_distance(self, x_1, x_2):
        """
        Calculates the euclidean distance of two
        vectors (lists or numpy arrays)

        Parameters
        x_1 : numpy array or list
        x_2 : numpy array or list
            Both are vectorized representations of an image, or the
            encoding of the image

        Returns
        type: float
            Euclidean distance between two vectors,
        float value.

        """
        if x_1.ndim > 1 or x_2.ndim > 1:
            x_1 = x_1.flatten()
            x_2 = x_2.flatten()
        return distance.euclidean(x_1, x_2)


    def c_loss(self, y, y_hat):
        """This one was copied fomr the repo but i intend to change it
        Maybe check out
        https://github.com/MGBergomi/siameseCNN
        """



    def pairs(self):
        """Gets the pairs of samples and assigns a similarity label"""
        pass



    #this will be changed to a class called Datasets that implements a more
    #elegant approach
    def load_data(self):
        """Imports relevant standard datasets"""
        if self.load_data == True:
            from keras.datasets import mnist
            mnist_dataset = mnist.load_data()
        return mnist_dataset

    #This one tries to mimic Le Cun's description of the model used
    #On the airplanes dataset..
    def model_1_NNet(self, input_shape):
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(20, activation='relu')(x)
        x = Dense(3, activation='relu')(x)
        return Model(input, x)



    def model_2_CNN(self, input_shape):
        pass


    def accuracy(self):
        pass

    def plot(self):
        pass
