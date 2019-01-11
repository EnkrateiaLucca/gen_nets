# import numpy as np
# from random import randint, random
# from operator import add
# from sklearn.datasets import load_iris
# import keras
# from sklearn.model_selection import train_test_split
#
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
#
# iris = load_iris()
# X = iris['data']
# y = iris['target']
# names = iris['target_names']
# feature_names = iris['feature_names']
#
# # One hot encoding
# enc = OneHotEncoder()
# Y = enc.fit_transform(y[:, np.newaxis]).toarray()
#
# # Scale data to have mean 0 and variance 1
# # which is importance for convergence of the neural network
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Split the data set into training and testing
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_scaled, Y, test_size=0.5, random_state=2)
#
# n_features = X.shape[1]
# n_classes = Y.shape[1]
#
#
#
#
#
# model = keras.models.Sequential([
#         keras.layers.Dense(5, activation="relu", input_dim = 4),
#         keras.layers.Dense(3, activation="softmax")
# ])
#
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics = ["accuracy"])
#
#
#
#
# model.fit(X_train, Y_train, epochs=1, verbose=1)
#
#
#
#
#
#
#
#
#
# def individual():
#     pass #the individuals
#
# def population():
#     pass #the actual population
#
# def fitness():
#     pass #fitness values for individuals in population
#
# def grade():
#     pass #average fitness value of a population
#
# def evolve():
#     pass #evolve function that serves as the guide


## TODO: do recombination of the best fit ones with less fit ones
