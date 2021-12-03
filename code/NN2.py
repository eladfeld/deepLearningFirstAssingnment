import numpy as np
import funcs
# from matplotlib import pyplot as plt
# import random
# from utils import *
# import scipy.io as sio
# from copy import deepcopy


class Layer:
    def __init__(self, in_dim, out_dim):
        self.weights = np.random.uniform(size=(in_dim, out_dim), low=-1.0, high=1.0 )
        self.bias= np.random.uniform(size=(out_dim, 1), low=-1.0, high=1.0 ) #todo: may need to switch dims of size param
        self.Y = np.zeros((out_dim, 1)) #todo: may need to switch dims order

class NN:
    def __init__(self, layer_dims, act=funcs.tanh, act_tag=funcs.tanh_tag, lr=0.01):
        self.act = act
        self.d_act = act_tag
        self.lr = lr
        self.num_layers = len(layer_dims) - 1

        self.layers = [None] * self.num_layers
        for i in range(0, self.num_layers):
            self.layers[i] = Layer(layer_dims[i], layer_dims[i+1])

    def predict(self, X):
        output = X

        #propogate data forward through the layers
        for i in range(0, self.num_layers):
            output = (output @ self.layers[i].weights)
            output += self.layers[i].bias.T
            f = funcs.softmax if (i == self.num_layers - 1) else self.act
            output = f(output)
            self.layers[i].Y = output

        return output        

    def learn(self, expected, data):
        pred = self.layers[-1].Y
        err = (pred - expected).T

        for i in range(2, self.num_layers + 1):
            d_f = funcs.grad_softmax if (i == 2) else self.d_act
            a = self.layers[self.num_layers - i].Y.T
            b = (err.T * d_f(self.layers[self.num_layers - i + 1].Y))#todo: transpose result, not just err?
            dW = -self.lr * (a @ b)
            dB = -self.lr * b
            # print(f"learn: a: {a.shape}, b: {b.shape}, bias: {self.layers[self.num_layers - i + 1].bias.shape}")

            err = self.layers[self.num_layers - i + 1].weights @ err
            self.layers[self.num_layers - i + 1].weights += dW
            self.layers[self.num_layers - i + 1].bias += dB.T

        a = data.T
        b = err.T
        dW = -self.lr * (a @ b)
        dB = -self.lr * b
        self.layers[0].weights += dW
        self.layers[0].bias += dB.T

    def train(self, inputs, labels):
        batch_size = 250
        batch_err = 0
        bad_preds = 0
        for i in range(len(inputs)):
            input = inputs[i][None, :]
            expected = labels[i][None, :]
            pred = self.predict(input)

            batch_err += get_error(pred, expected)
            if not pred_is_correct(pred, expected):
                bad_preds += 1

            self.learn(expected, input)

            if (i + 1) % batch_size == 0:
                print(f"batch #{int(i/batch_size)}:\twrongs: {bad_preds}/{batch_size}\terr: {batch_err}")
                batch_err = 0
                bad_preds = 0




            
def pred_is_correct(pred, real):
    return (np.argmax(pred) == np.argmax(real))


def get_error(pred, real):
    output = pred - real
    output = output * output
    output = np.sum(output)
    return output


def foo(v):
    v1, v2, v3 = v[0]
    if (v1 + v2 > v3):
        return np.array([[0, 1]])
    else:
        return np.array([[1, 0]])


def check():
    in_dim = 3
    out_dim = 2
    nn = NN(layer_dims=[in_dim, 4, 3, out_dim], act=funcs.tanh, act_tag=funcs.tanh_tag)

    train_size = 500000
    print_every_n = 1000
    f = foo

    err = 0
    for i in range(1, train_size + 1):
        input = np.random.uniform(0, 1, (1, in_dim))
        expected = f(input)

        pred = nn.predict(input)
        err += get_error(pred, expected)
        if i%print_every_n == 0:
            print(f"avg_err: {err / print_every_n}")
            err = 0
            
        nn.learn(expected, input)


# check()