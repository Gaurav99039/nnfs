import numpy as np
from nnfs.datasets import spiral_data
class Layer_Dense:

     #layer intialization
    def __init__(self,n_inputs,n_neurons):
        #initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        #calculate the output values from inputs biases and weights
        self.outputs = np.dot(inputs,self.weights) + self.biases

X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,3)
dense1.forward(X)
print(dense1.outputs[:5])