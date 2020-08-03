import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = plt.imread('/Users/andrewxue/cat.png')

class SingleNodeNeuralNetwork:

    def init(self, X, Y):
        self.weights = np.random.randn(1, X.shape[0])*0.01
        self.bias = np.zeros((1, 1))

    def forward_propagation(self, X):
        Z = np.dot(self.weights, X)+self.bias
        A = 1/(1+np.exp(-Z))
        return A

    def backward_propagation(self, A, X, Y):
        dZ = A-Y
        dW = np.dot(self.weights, X)
        dB = dZ
        return dW, dB

    def train(self, X, Y, iterations, learning_rate):
        m = X.shape[0]
        X = X.flatten().reshape(X.shape[0], -1).T
        self.init(X, Y)
        for i in range(iterations):
            print("iter:", i)
            A = self.forward_propagation(X)
            print(A)
            C = -(np.sum(np.dot(Y, A))+np.sum(np.dot(1-Y, 1-A)))/m
            print(C)
            dW, dB = self.backward_propagation(A, X, Y)
            self.weights = self.weights-learning_rate*dW
            self.bias = self.bias-learning_rate*dB

net = SingleNodeNeuralNetwork()
net.train(np.array([img]), np.array([1]), 10, 0.01)
