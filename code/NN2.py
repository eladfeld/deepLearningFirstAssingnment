import numpy as np
from matplotlib import pyplot as plt
import random
from utils import *
import scipy.io as sio
from copy import deepcopy

LR = 0.01

class Layer:
    def __init__(self, in_dim, out_dim):
        self.weights = np.random.uniform(size=(out_dim, in_dim), low=-1.0, high=1.0 )
        self.bias= np.random.uniform(size=(out_dim, 1), low=-1.0, high=1.0 ) #todo: may need to switch dims of size param
        self.Y = np.zeros((out_dim, 1)) #todo: may need to switch dims order

class NN:
    def __init__(self, layer_dims, act=tanh, act_grad=tanh_grad):
        self.act = act
        self.act_grad = act_grad
        self.num_layers = len(layer_dims) - 1

        self.layers = [None] * self.num_layers
        for i in range(0, self.num_layers):
            self.layers[i] = Layer(layer_dims[i], layer_dims[i+1])

    def predict(self, X):
        output = X
        for i in range(0, self.num_layers):
            output = (self.layers[i].weights @ output) + self.layers[i].bias[:,None]
            f = softmax if (i == self.num_layers) else self.act
            output = f(output)
            self.layers[i].Y = output

        return output        

    #todo: complete...
    #returns gradient w.r.t. W,B of last prediction
    def grad_layer(self,layer_index, X, C):
        """

                Args:
                    layer_index: index for layer
                    X: input data
                    C: one-hot class vector

                Returns: gradient w.t.r to Weights and bias

                """
        if layer_index==(len(self.layers)-1):#In case we want to compute the last layer
            W=self.layers[-1].weights
            layer_output=self.forwardpassToHiddenLayer(X,len(self.layers)-1)
            bias_grad = softmax_grad(softmax(W @ layer_output + self.layers[-1].bias[:, None]),
                                     np.ones((C.shape[1], 1)).T, C)

            return softmax_grad(  softmax((W @ layer_output) + self.layers[-1].bias[:,None]) , layer_output, C  ),bias_grad

        layer = self.layers[layer_index]#
        m = C.shape[1]
        W = self.layers[-1].weights
        layer_output = layer.Y#
        A = softmax(W @ layer_output + self.layers[-1].bias[:, None])
        grad=(1/m) *  W.T @(A - C )

        i=len(self.layers)-2
        while i>layer_index:#backpropegation, compute the X deveratives of all the layers after the layer at layer_index
            W=self.layers[i].weights
            bias=self.layers[i].bias
            layer_output=self.forwardpassToHiddenLayer(X, i)#get the X values fed as input to the layer
            activation_function_grad=self.layers[i].activationfuncgrad
            grad=np.transpose(W) @ np.multiply(activation_function_grad(W @ layer_output + bias[:,None]),grad)

            i=i-1
        #compute the jacobian of all weights of the layer at position layer_index
        W=self.layers[layer_index].weights
        bias=self.layers[layer_index].bias
        activation_function_grad=self.layers[layer_index].activationfuncgrad
        layer_output=self.forwardpassToHiddenLayer(X, layer_index)
        gradW= np.multiply(activation_function_grad((W @ layer_output) + bias[:,None]), grad  ) @ np.transpose(layer_output)
        gradBias = np.multiply(activation_function_grad(W @ layer_output + bias[:, None]), grad) @ np.ones(
            (C.shape[1], 1))

        return gradW,gradBias

    #todo: complete...
    def train(self, data, test, validation=None, lr=0.01, num_epochs=50, batch_size=25, lr_reducer=False):
        """

        Args:
            data: input data for training
            test: classification for each data row
            validation: data used for validation of the network, not training
            learning_rate: the SGD coefficient
            epoch: numbers of itartion on all the data
            batch_size: size of batch per iteration
            lr_reducer: if True -> the network will reduce the learning rate if the loss function is increasing.

        Returns:history of the train and validation results.

        """
        train_loss = []
        train_acuracy = []
        validation_loss = []
        validation_acuracy = []

        num_samples = len(data[0])

        for j in range(num_epochs):
            evall = self.predict(data)
            f = cross_entropy(evall, test)
            prediction = np.argmax(evall, axis=0)
            real = np.argmax(test, axis=0)
            accuracy = np.mean(prediction == real)
            train_acuracy.append(accuracy)
            train_loss.append(f)
            # if len(train_loss) > 3 and train_loss[-1] > train_loss[-2] and lr_reducer:
            #     learning_rate = learning_rate ** 2
            for i in range(0, num_samples, batch_size):

                evall = self.predict(data[:, i:i + batch_size])
                f = cross_entropy(evall, test[:, i:i + batch_size])
                prediction = np.argmax(evall, axis=0)
                real = np.argmax(test[:, i:i + batch_size], axis=0)
                accuracy = np.mean(prediction == real)
                for h in range(0, len(self.layers)):
                    gradW, gradB = self.grad_layer(h, data[:, i:i + batch_size], test[:, i:i + batch_size])

                    self.layers[h].weights = self.layers[h].weights - gradW * lr
                    self.layers[h].bias = (self.layers[h].bias - gradB.T * lr).reshape(
                        self.layers[h].bias.shape)
                    # bias_grad = softmax_grad(
                    #    softmax((NN.layers[h].weights @ data[:, i:i + batch_size]) + NN.layers[-1].bias[:, None]),
                    #    np.ones((test[:, i:i + batch_size].shape[1], 1)).T, test[:, i:i + batch_size])
                    #
                    # results.append(NN.layers[h].weights)

            if validation is not None:
                evall = self.predict(validation[0])
                # f = cross_entropy(evall, validation[1])
                prediction = np.argmax(evall, axis=0)
                real = np.argmax(validation[1], axis=0)
                accuracy = np.mean(prediction == real)
                validation_acuracy.append(accuracy)
                validation_loss.append(f)
                # print(accuracy)
        return {'train': {'loss': train_loss, 'auc': train_acuracy},
                'test': {'loss': validation_loss, 'auc': validation_acuracy}}