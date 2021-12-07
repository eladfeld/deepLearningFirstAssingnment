import numpy as np
import funcs
from utils import fs, unison_shuffled_copies
# from matplotlib import pyplot as plt
# import random
# from utils import *
# import scipy.io as sio
# from copy import deepcopy
LR = 2
LR_DECAY = .9999
MIN_LR = 0.000000001
DEBUG = False

class Layer:
    def __init__(self, in_dim, out_dim):
        np.random.seed(42)
        self.weights = np.random.normal(size=(in_dim, out_dim), loc=0.0, scale=1.0 )
        self.bias= np.random.normal(size=(out_dim, 1), loc=0.0, scale=1.0 )
        self.Y = np.zeros((out_dim, 1)) #todo: may need to switch dims order
        self.X = None
        self.df_dtheta = None
    
    def dy_dw_t_v(self, v, act_tag):
        W, b, x = self.weights, self.b, self.X

        output = act_tag((W @ x) + b)
        output = output * v
        output = output * x.T



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
        my_print("\n\n********** Predict() **********")
        my_print(f"input: {X}")
        output = X

        #propogate data forward through the layers
        for i in range(0, self.num_layers):
            self.layers[i].X = output

            my_print(f"\nlayer: {i}")
            my_print(f"\tW: {self.layers[i].weights}")

            output = (output @ self.layers[i].weights)

            my_print(f"\toutput: {output}")

            #output += self.layers[i].bias.T
            f = funcs.softmax if (i == self.num_layers - 1) else self.act

            f_str = "softmax" if (i == self.num_layers - 1) else "self.act"

            output = f(output)
            
            my_print(f"\t\t{f_str}(output): {output}")
            self.layers[i].Y = output

        return output        

    def learn(self, expected, data):
        my_print("\n\n********** Learn() **********")
        pred = self.layers[-1].Y
        my_print(f"input: {data}\n")
        my_print(f"pred: {pred}\n")
        my_print(f"real: {expected}\n")
        err = (pred - expected).T
        
        for i in range(2, self.num_layers + 2):
            my_print(f"\ni: {i}")
            my_print(f"\terr: {err}")

            d_f = funcs.grad_softmax_old if (i == 2) else self.d_act
            d_f_str = "grad_softmax" if (i == 2) else "self.d_act"
            my_print(f"\td_f: {d_f_str}")

            input = self.layers[self.num_layers - i].Y if (i < self.num_layers + 1) else data
            output = self.layers[self.num_layers - i + 1].Y

            my_print(f"\tinput: {input}\n")
            my_print(f"\toutput: {output}\n")

            a = input.T
            b = (err.T * d_f(output))

            my_print(f"\ta: {a.shape}, b: {b.shape}")
            dW = -self.lr * (err @ input)#(a @ b)
            dB = -(self.lr) * np.mean(b, axis=0, keepdims=True)

            my_print(f"\tdW:\n{dW}")

            err = self.layers[self.num_layers - i + 1].weights @ err

            my_print(f"\tW before update:\n{self.layers[self.num_layers - i + 1].weights}")
            self.layers[self.num_layers - i + 1].weights += dW.T
            my_print(f"\tW after update:\n{self.layers[self.num_layers - i + 1].weights}")
            #self.layers[self.num_layers - i + 1].bias += dB.T


        self.lr = max(LR_DECAY * self.lr, MIN_LR)

    def train(self, inputs, labels, mini_batch_size, batch_size, num_epochs=1):
        batch_err = 0
        num_correct, total = 0, 0
        p_stats = np.zeros((labels.shape[1]))
        r_stats = np.zeros((labels.shape[1]))
        epoch_acc = 0

        for epoch in range(num_epochs):
            inputs, labels = unison_shuffled_copies(inputs, labels)
            print(f"---------- Epoch #{epoch} ----------")
    
            for i in range(0, len(inputs), mini_batch_size):
                input = inputs[i:i+mini_batch_size]
                expected = labels[i:i+mini_batch_size]
                pred = self.predict(input)

                batch_err += get_error(pred, expected)
                num_correct_i, total_i = accuracy(pred, expected)
                num_correct += num_correct_i
                total += total_i
                epoch_acc += num_correct_i

                p_stats_i, r_stats_i = stats(pred, expected)
                p_stats += p_stats_i
                r_stats += r_stats_i
                    

                self.learn(expected, input)

                if (i + mini_batch_size) % batch_size == 0:
                    print(f"{int(i/batch_size)}\tlr: {fs(self.lr)}\terr: {fs(batch_err/batch_size)}\tacc: {num_correct}/{total}")#\tps: {p_stats}\trs: {r_stats}")
                    batch_err = 0
                    num_correct, total = 0, 0
                    p_stats = np.zeros((labels.shape[1]))
                    r_stats = np.zeros((labels.shape[1]))

            
            print(f"epoch acc: {epoch_acc} / {len(inputs)}\t({int((epoch_acc * 100)/len(inputs))}%)")
            epoch_acc = 0
            self.lr = LR

def accuracy(pred, real):
    num_correct = 0
    total = len(pred)
    for i in range(total):
        if (np.argmax(pred[i]) == np.argmax(real[i])):
            num_correct += 1
    return num_correct, total

def print_pred_real_acc(pred, real):
    print("*************")
    print("pred:")
    print(pred)
    print("\nreal:")
    print(real)
    print(f"acc: {accuracy(pred, real)}")
    print("*************")

def stats(preds, reals):
    p_stats = [0] * preds.shape[1]
    r_stats = [0] * reals.shape[1]

    for i in range(preds.shape[0]):
        pred = np.argmax(preds[i])
        p_stats[pred] += 1

        real = np.argmax(reals[i])
        r_stats[real] += 1
    
    return np.array(p_stats), np.array(r_stats)

            



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

def my_print(s):
    if DEBUG:
        print(s)