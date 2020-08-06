import numpy as np
from matplotlib import pyplot as plt
import os
import h5py

class MultiLayerNeuralNetwork:

	def init(self, X, Y, dimensions):
		self.weights = [0]*self.L
		self.biases = [0]*self.L 
		self.cache = {}
		for i in range(1, self.L):
			self.weights[i] = np.random.randn(dimensions[i], dimensions[i-1])*0.01
			self.biases[i] = np.random.randn(dimensions[i], 1)
	
	def sigmoid(self, Z):
		A = 1/(1+np.exp(-Z))
		return A

	def relu(self, Z):
		A = np.maximum(0, Z)
		return A

	def forward_propagation(self, X):
		A = X
		self.cache["A0"] = A
		for i in range(1, self.L-1):
			Z = np.dot(self.weights[i], A)+self.biases[i] # (d_i, m) Vector
			self.cache["Z"+str(i)] = Z
			A = self.relu(Z) # (d_i, m) Vector
			self.cache["A"+str(i)] = A
		Z = np.dot(self.weights[self.L-1], A)+self.biases[self.L-1]	
		self.cache["Z"+str(self.L-1)] = Z
		A = self.sigmoid(Z)
		self.cache["A"+str(self.L-1)] = A
		return A

	def sigmoid_backward(self, dA, Z):
		s = 1/(1+np.exp(-Z))
		dZ = dA*s*(1-s) # (d_i, m) Vector 
		return dZ
	
	def relu_backward(self, dA, Z):
		dZ = np.array(dA, copy=True) # (d_i, m) Vector
		dZ[Z <=0 ] = 0
		return dZ

	def backward_propagation(self, dA, lambd, learning_rate):
		dZ = self.sigmoid_backward(dA, self.cache["Z"+str(self.L-1)])
		dW = np.dot(dZ, self.cache["A"+str(self.L-2)].T)/self.m+lambd*self.weights[self.L-1]/self.m # (d_i, d_i-1) Vector
		dB = np.sum(dZ, axis=1, keepdims=True)/self.m # (d_i, 1) Vector 
		dA = np.dot(self.weights[self.L-1].T, dZ) # (d_i-1, m) Vector
		self.weights[self.L-1] = self.weights[self.L-1]-learning_rate*dW
		self.biases[self.L-1] = self.biases[self.L-1]-learning_rate*dB	
		for i in range(self.L-2, 0, -1):
			dZ = self.relu_backward(dA, self.cache["Z"+str(i)])
			dW = np.dot(dZ, self.cache["A"+str(i-1)].T)/self.m # (d_i, d_i-1) Vector

			dB = np.sum(dZ, axis=1, keepdims=True)/self.m # (d_i, 1) Vector
			
			dA = np.dot(self.weights[i].T, dZ) # (d_i-1, m) Vector

			self.weights[i] = self.weights[i]-learning_rate*dW
			self.biases[i] = self.biases[i]-learning_rate*dB

	def accuracy(self, A, Y):
		correct = 0
		for i in range(self.m):
			if (round(A[0][i]) == Y[0][i]):
				correct += 1
		return correct/self.m	

	def regularization(self, lambd):
		tot = 0
		for i in range(1, self.L):
			tot += np.sum(self.weights[i]*self.weights[i])
		return tot*lambd/(2*self.m)

	def train(self, X, Y, dimensions, iterations, learning_rate, lambd):
		np.random.seed()
		self.m = X.shape[0]
		print(self.m)
		X = X.reshape(X.shape[0], -1).T # (n_x, m) Vector
		self.L = len(dimensions)
		self.init(X, Y, dimensions)
		for i in range(iterations):
			A = self.forward_propagation(X) # (d_l, m) Vector
			C = -np.sum(np.multiply(Y, np.log(A))+np.multiply(1-Y, np.log(1-A)))/self.m+self.regularization(lambd)
			self.backward_propagation(-(np.divide(Y, A)-np.divide(1-Y, 1-A)), lambd, learning_rate)	
			if ((i+1)%100==0):
				print("epoch:", i)
				print("cost:", C)
				print("accuracy:", self.accuracy(A, Y))
				print('-'*10)
		
	def predict(self, X):
		Y = np.zeros((1, X.shape[0]))
		self.m = X.shape[0]
		X = X.reshape(X.shape[0], -1).T # (n_x, m) Vector
		A = self.forward_propagation(X)
		for i in range(self.m):
			Y[0][i] = round(A[0][i])
		return Y, A
	

net = MultiLayerNeuralNetwork()

f = h5py.File(os.getcwd()+'/data/train.h5', 'r')

X = np.array(f['train_set_x'])/255
Y = np.array(f['train_set_y']).reshape((1, X.shape[0]))

net.train(X, Y, [int(np.prod(X.shape)/X.shape[0]), 50, 1], 1500, 0.01, 1)

f = h5py.File(os.getcwd()+'/data/test.h5', 'r')

X = np.array(f['test_set_x'])/255
Y = np.array(f['test_set_y']).reshape((1, X.shape[0]))
y, a = net.predict(X)

correct = 0
for i in range(X.shape[0]):
	if y[0][i] == Y[0][i]:
		correct += 1

print("test result:", correct/X.shape[0])
