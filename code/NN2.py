import numpy as np
# from matplotlib import pyplot as plt
# import random
# from utils import *
# import scipy.io as sio
# from copy import deepcopy

LR = 0.01

class Layer:
    def __init__(self, in_dim, out_dim):
        self.weights = np.random.uniform(size=(in_dim, out_dim), low=-1.0, high=1.0 )
        self.bias= np.random.uniform(size=(out_dim, 1), low=-1.0, high=1.0 ) #todo: may need to switch dims of size param
        self.Y = np.zeros((out_dim, 1)) #todo: may need to switch dims order

class NN:
    def __init__(self, layer_dims, act=None, act_grad=None):
        self.act = act
        self.act_grad = act_grad
        self.num_layers = len(layer_dims) - 1

        self.layers = [None] * self.num_layers
        for i in range(0, self.num_layers):
            self.layers[i] = Layer(layer_dims[i], layer_dims[i+1])

    def predict(self, X):
        output = X
        for i in range(0, self.num_layers):
            #print(f"i: {i},out: {output.shape}, w: {self.layers[i].weights.shape}")
            output = (output @ self.layers[i].weights)
            output += self.layers[i].bias.T
            #f = softmax if (i == self.num_layers) else self.act
            f = lambda x: x
            output = f(output)
            self.layers[i].Y = output

        return output        

    def learn(self, expected, data):
        pred = self.layers[-1].Y
        err = (pred - expected).T

        for i in range(2, self.num_layers + 1):
            a = self.layers[self.num_layers - i].Y.T
            b = err.T
            dW = -LR * (a @ b)
            dB = -LR * b
            # print(f"learn: a: {a.shape}, b: {b.shape}, bias: {self.layers[self.num_layers - i + 1].bias.shape}")

            err = self.layers[self.num_layers - i + 1].weights @ err
            self.layers[self.num_layers - i + 1].weights += dW
            self.layers[self.num_layers - i + 1].bias += dB.T

        a = data.T
        b = err.T
        #print(f"a: {a.shape}, b: {b.shape}")
        dW = -LR * (a @ b)
        dB = -LR * b
        #print(f"dW: {dW.shape}, W: {self.layers[0].weights.shape}")
        self.layers[0].weights += dW
        self.layers[0].bias += dB.T


            


def check():
    in_dim = 5
    out_dim = 2
    nn = NN(layer_dims=[in_dim, 4, 3, out_dim])

    # for layer in nn.layers:
    #     print(layer.weights.shape)

    for i in range(1,200):
        input = np.ones((1, in_dim)) * (0.01 * i)
        expected = np.ones((1, out_dim)) * (0.01 * i)

        pred = nn.predict(input)
        if i%10 == 0:
            print(f"exp: {0.01 * i},\tgot: {pred}")
        nn.learn(expected, input)

        # print(pred)
        # for i in range(0, 10):
        #     nn.learn(expected, input)
        #     pred = nn.predict(input)
        #     print(pred)




check()