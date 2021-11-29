import numpy as np
import matplotlib.pyplot as plt


class Softmax:

    def __init__(self, num_inputs, num_class):
        self.W = (1 / num_inputs) * np.random.randn(num_inputs, num_class)

    def softmax(self, xTw):
        e_x = np.exp(xTw - np.max(xTw))
        return e_x / np.sum(e_x, keepdims=True, axis=1)

    def forward(self, X):
        return self.softmax(X.T.dot(self.W))

    def predict(self, X):
        # TODO: maybe reshape!!
        return np.argmax(self.forward(X), axis=1)

    def cross_entropy(self, output, labels):
        m = output.shape[0]
        entropized = np.log(output) * labels.T
        sumed = np.sum(entropized, axis=1, keepdims=True)
        return np.sum(sumed, axis=0) / -m

    def cross_entropy_grad_wrt_weights(self, X, output, labels):
        pass

    def cross_entropy_grad_wrt_inputs(self, X, labels):
        pass

    def grad_test_wrt_w(self):
        F = lambda x: 0.5 * np.dot(x, x)
        g_F = lambda x: x
        n = 20
        x = np.random.randn(n)
        d = np.random.randn(n)
        epsilon = 0.1
        F0 = F(x)
        g0 = g_F(x)
        y0 = np.zeros(8)
        y1 = np.zeros(8)
        print("k\terror  oreder 1\t\t error order 2")
        for k in range(1, 9):
            epsk = epsilon * (0.5 ** k)
            Fk = F(x + epsk * d)
            F1 = F0 + epsk * np.dot(g0, d)
            y0[k] = np.abs(Fk - F0)
            y1[k] = np.abs(Fk - F1)
            print(f'{k} \t {y0[k]} \t {y1[k]}')
        return y0, y1

    def sgd(self, x):
        return np.sgd(x)