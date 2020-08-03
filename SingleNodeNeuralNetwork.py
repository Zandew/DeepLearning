import numpy as np
import os
import h5py

class SingleNodeNeuralNetwork:	

	def init(self, X, Y):
		self.weights = np.random.randn(1, X.shape[0])*0.01 # (1, n_x) Row Vector 
		self.bias = 0

	def forward_propagation(self, X):
		Z = np.dot(self.weights, X)+self.bias # (1, m) Row Vector
		A = 1/(1+np.exp(-Z))
		return A # return (1, m) vector

	def backward_propagation(self, A, X, Y):
		dW = np.dot(A-Y, X.T)/self.m # (1, n_x) Row Vector  
		dB = np.sum(A-Y)/self.m # Scalar
		return dW, dB

	def accuracy(self, A, Y):
		correct = 0
		for i in range(self.m):
			if round(A[0][i]) == Y[0][i]:
				correct += 1
		return correct/self.m

	def train(self, X, Y, iterations, learning_rate):
		self.m = X.shape[0]
		X = X.reshape(X.shape[0], -1).T # (n_x, m) Vector
		self.init(X, Y)
		for i in range(iterations):
			print("iter:", i)
			A = self.forward_propagation(X)		
			C = -np.sum(np.multiply(Y, np.log(A))+np.multiply(1-Y, np.log(1-A)))/self.m
			print("cost:", C)
			dW, dB = self.backward_propagation(A, X, Y)		
			self.weights = self.weights-learning_rate*dW
			self.bias = self.bias-learning_rate*dB
			print("accuracy:", self.accuracy(A, Y))	
			print("-"*10)
	
	def predict(self, X):
		Y = np.zeros((1, X.shape[0]))	
		self.m = X.shape[0]
		X = X.reshape(X.shape[0], -1).T # (n_x, m) Vector
		A = self.forward_propagation(X)
		for i in range(self.m):
			Y[0][i] = round(A[0][i])
		return Y	

net = SingleNodeNeuralNetwork()

f = h5py.File(os.getcwd()+'/train.h5', 'r')

X = np.array(f['train_set_x'])/255
Y = np.array(f['train_set_y']).reshape((1, X.shape[0]))
net.train(np.array(X), np.array(Y), 5000, 0.005)

f = h5py.File(os.getcwd()+'/test.h5', 'r')

X = np.array(f['test_set_x'])/255
Y = np.array(f['test_set_y']).reshape((1, X.shape[0]))
y = net.predict(X)

correct = 0
for i in range(X.shape[0]):
    if y[0][i] == Y[0][i]:
        correct += 1

print("test result:", correct/X.shape[0])

