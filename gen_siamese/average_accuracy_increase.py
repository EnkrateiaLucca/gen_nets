import numpy as np
import matplotlib.pyplot as plt

mnist_results = [99.28, 99.88, 99.90, 99.88, 99.90, 99.91, 99.91, 99.91, 99.85, 99.90]
fashion_mnist_results = [98.28, 98.96, 99.10, 99.21, 99.23, 99.24,99.21, 99.24,99.10, 99.26]
cifar10_results = [68.24, 71.27, 71.33, 71.18, 72.71, 71.98, 72.42, 71.67, 73.34, 72.99]
cifar100_results = [69.66, 70.86, 76.24, 77.26, 77.41, 77.56, 76.11, 78.03, 77.64, 77.60]

mnist_increase = []
fashion_mnist_increase = []
cifar10_increase = []
cifar100_increase = []

for i in range(1,len(mnist_results)):
    mnist_increase.append(mnist_results[i]- mnist_results[i-1])

plt.scatter(list(range(9)),mnist_increase, color = 'r')
plt.title("Average increase value {}".format(np.mean(mnist_increase)))
plt.xlabel(xlabel='Generations')
plt.ylabel(ylabel='Increase')
plt.savefig("Mnist_increase_plot.png")
plt.show()

for i in range(1,len(fashion_mnist_results)):
    fashion_mnist_increase.append(fashion_mnist_results[i]- fashion_mnist_results[i-1])

plt.scatter(list(range(9)),fashion_mnist_increase, color = 'r')
plt.title("Average increase value {}".format(np.mean(fashion_mnist_increase)))
plt.xlabel(xlabel='Generations')
plt.ylabel(ylabel='Increase')
plt.savefig("Fashion_mnist_increase_plot.png")
plt.show()

for i in range(1,len(cifar10_results)):
    cifar10_increase.append(cifar10_results[i]- cifar10_results[i-1])
plt.scatter(list(range(9)),cifar10_increase, color = 'r')
plt.title("Average increase value {}".format(np.mean(cifar10_increase)))
plt.xlabel(xlabel='Generations')
plt.ylabel(ylabel='Increase')
plt.savefig("Cifar10_increase_plot.png")
plt.show()

for i in range(1,len(cifar100_results)):
    cifar100_increase.append(cifar100_results[i]- cifar100_results[i-1])
plt.scatter(list(range(9)),cifar100_increase, color = 'r')
plt.title("Average increase value {}".format(np.mean(cifar100_increase)))
plt.xlabel(xlabel='Generations')
plt.ylabel(ylabel='Increase')
plt.savefig("Cifar100_increase_plot.png")
plt.show()

print("Mnist average increase:")
print(np.mean(mnist_increase))
print("*"*100)

print("Fashion mnist average increase:")
print(np.mean(fashion_mnist_increase))
print("*"*100)

print("Cifar10 average increase:")
print(np.mean(cifar10_increase))
print("*"*100)

print("Cifar100 average increase:")
print(np.mean(cifar100_increase))
print("*"*100)
