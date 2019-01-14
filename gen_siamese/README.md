# Evolving a siamese network (with a contrastive loss function) with a genetic algorithm

This code implements a genetic algorithm that evolves a siamese network (with a contrastive loss function) to aid on hyperparameters search. 

It uses keras for the training and validation

The datasets used were: mnist, fashion mnist, cifar10, cifar100. 

## To run

To run the genetic algorithm:

'python main.py'

Dataset options: mnist, fashion_mnist, cifar10, cifar100.
(just set the dataset variable on main.py to any of these datasets)

## Credits
The genetic algorithm code is based on a combination of posts and repositories, mainly: 
    - blog post: https://lethain.com/genetic-algorithms-cool-name-damn-simple/
    - https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
    - https://github.com/harvitronix/neural-network-genetic-algorithm
    - https://github.com/jliphard/DeepEvolve

## License

MIT


