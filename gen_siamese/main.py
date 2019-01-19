"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

# Setup logging.

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log_{}.txt'.format('mnist')
)




def train_networks(networks, dataset, curr_gen_num):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    net_num = 0
    pbar = tqdm(total=len(networks))
    acc_list = []
    print("This is the total size of the current network: {}".format(len(networks)))
    for network in networks:
        network.train(dataset, curr_gen_num)
        plt.plot(network.ind_acc)
        acc_list.append(network.ind_acc)
        pbar.update(1)


    pbar.close()
    averages = [np.around(np.mean(acc), 2) for acc in acc_list]
    plt.title("generation {} average accuracy: {}".format(curr_gen_num,np.mean(averages)))
    plt.xlabel(xlabel='Epochs')
    plt.ylabel(ylabel='Accuracy')
    plt.savefig('acc')
    plt.close()
    os.chdir('../')

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    acc_list = []
    for network in networks:
        total_accuracy += network.accuracy
        acc_list.append(network.ind_acc)
    avg_acc = total_accuracy / len(networks)

    return avg_acc, acc_list

def generate(generations, population, nn_param_choices, dataset):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    print("The size of the population is {}".format(population))
    networks = optimizer.create_population(population)
    print("These are all the networks created for this population:")
    print(networks)

    # Evolve the generation.
    for i in range(generations):
        os.mkdir('./training_session_gen_{}'.format(i+1))
        print("Doing generation: {}".format(i+1))
        os.chdir('./training_session_gen_{}'.format(i+1))
        logging.info("***Doing generation {} of {}***".format(i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, dataset, i)

        # Get the average accuracy for this generation.
        average_accuracy, acc_list = get_average_accuracy(networks)
        print("This is the average_accuracy: {}".format(average_accuracy))


        # Print out the average accuracy each generation.
        logging.info("Generation average: {}".format(average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def plot_gen_training(avg_acc, savefig = False, show=False):
    """Plots the training accuracy and loss"""
    for net in avg_acc:
        plt.plot(net, c="b")
        plt.title("Accuracy")
    if savefig:
        plt.savefig("training.png")
    if show:
        plt.show()

def main(dataset):
    """Evolve a network."""
    generations = 10  # Number of times to evole the population.
    population = 10 # Number of networks in each generation.
    dataset = dataset

    nn_param_choices = {
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax'],
    }

    logging.info("***Evolving {} generations with population {}***".format(generations, population))

    generate(generations, population, nn_param_choices, dataset)

if __name__ == '__main__':
    main('mnist')
