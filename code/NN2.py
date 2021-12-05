import numpy as np
import funcs
from utils import fs
# from matplotlib import pyplot as plt
# import random
# from utils import *
# import scipy.io as sio
# from copy import deepcopy
LR = 0.15
LR_DECAY = .9999
MIN_LR = 0.025

class Layer:
    def __init__(self, in_dim, out_dim):
        np.random.seed(42)
        self.weights = np.random.uniform(size=(in_dim, out_dim), low=-1.0, high=1.0 )
        self.bias= np.random.uniform(size=(out_dim, 1), low=-1.0, high=1.0 ) #todo: may need to switch dims of size param
        self.Y = np.zeros((out_dim, 1)) #todo: may need to switch dims order

class NN:
    def __init__(self, layer_dims, act=funcs.tanh, act_tag=funcs.tanh_tag, lr=LR):
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
            #output += self.layers[i].bias.T
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
            #print(f"a: {a.shape, }b: {b.shape}, meanB0: {np.mean(b, axis=0, keepdims=True).shape}")
            dW = -self.lr * (a @ b)
            dB = -(self.lr / 10) * np.mean(b, axis=0, keepdims=True)
            #print(f"learn: a: {a.shape}, b: {b.shape}, dW: {dW.shape}, dB: {dB.shape}")

            err = self.layers[self.num_layers - i + 1].weights @ err
            self.layers[self.num_layers - i + 1].weights += dW
            self.layers[self.num_layers - i + 1].bias += dB.T

        a = data.T
        b = err.T * self.d_act(self.layers[0].Y)
        dW = -self.lr * (a @ b)
        dB = -(self.lr / 10) * np.mean(b, axis=0, keepdims=True)
        self.layers[0].weights += dW
        self.layers[0].bias += dB.T
        self.lr = max(LR_DECAY * self.lr, MIN_LR)

    def train(self, inputs, labels, mini_batch_size, batch_size):
        batch_err = 0
        num_correct, total = 0, 0
        p_stats = np.zeros((labels.shape[1]))
        r_stats = np.zeros((labels.shape[1]))
        
        for i in range(0, len(inputs), mini_batch_size):
            input = inputs[i:i+mini_batch_size]
            expected = labels[i:i+mini_batch_size]
            pred = self.predict(input)

            batch_err += get_error(pred, expected)
            num_correct_i, total_i = accuracy(pred, expected)
            num_correct += num_correct_i
            total += total_i

            p_stats_i, r_stats_i = stats(pred, expected)
            p_stats += p_stats_i
            r_stats += r_stats_i

            self.learn(expected, input)

            if (i + mini_batch_size) % batch_size == 0:
                # print(f"batch #{int(i/batch_size)}:\twrongs: {bad_preds}/{batch_size}\terr: {batch_err}")
                # print(f"batch #{int(i/batch_size)}:\terr: {batch_err/batch_size}")
                print(f"{int(i/batch_size)}\tlr: {fs(self.lr)}\terr: {fs(batch_err/batch_size)}\tacc: {num_correct}/{total}\tps: {p_stats}\trs: {r_stats}")
                batch_err = 0
                num_correct, total = 0, 0
                p_stats = np.zeros((labels.shape[1]))
                r_stats = np.zeros((labels.shape[1]))
            



def stats(preds, reals):
    p_stats = [0] * preds.shape[1]
    r_stats = [0] * reals.shape[1]

    for i in range(preds.shape[0]):
        pred = np.argmax(preds[i])
        p_stats[pred] += 1

        real = np.argmax(reals[i])
        r_stats[real] += 1
    
    return np.array(p_stats), np.array(r_stats)

            
def accuracy(pred, real):
    num_correct = 0
    total = len(pred)
    for i in range(total):
        if (np.argmax(pred[i]) == np.argmax(real[i])):
            num_correct += 1
    return num_correct, total


def get_error(pred, real):
    output = pred - real
    output = output * output
    output = np.sum(np.sum(output))
    # print(f"pred: {pred}\nreal: {real}\nerr: {output}")
    return output


def foo(v):
    v1, v2, v3 = v[0]
    if (v1 + v2 > v3):
        return np.array([[0, 1]])
    else:
        return np.array([[1, 0]])

