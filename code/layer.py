import numpy as np

LR = 0.01

class Layer:

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = np.ones((in_dim, out_dim))
        self.B = np.zeros((out_dim, 1))

        self.X = np.zeros((in_dim, 1))
        self.output = np.zeros((out_dim, 1))

    def forward(self, X):
        self.X = X
        self.output = (X.T @ self.W) + self.B.T
        return self.output

    def backward(self, desired):
        # print('output shape: ', self.output.shape)
        # print('desired shape: ', desired.shape)
        err = desired - self.output.T
        # print('err shape: ', err.shape)
        # print('X shape: ', self.X.shape)
        dW = LR * (self.X @ err.T)
        self.W = self.W + dW

        dB = LR * ()
        return self.W @ err

#simple nn with 1 in layer, 1 hidden layer, 1 output layer
class NNSimple:

    def __init__(self, in_dim, hid_dim, out_dim):
        self.l1 = Layer(in_dim, hid_dim)
        self.l2 = Layer(hid_dim, out_dim)
        #self.layers = [l1, l2]

    def predict(self, input):
        y1 = self.l1.forward(input)
        y2 = self.l2.forward(y1.T)
        return y2

    def learn(self, desired):
        e2 = self.l2.backward(desired)
        e1 = self.l1.backward(e2)
        return e1





def test_layer():
    layer = Layer(3, 2)
    input = np.ones((3, 1))
    desired = np.ones((2, 1))
    output = layer.forward(input)
    print(output)

    #try again and again
    for i in range(0, 10):
        layer.backward(desired)
        output = layer.forward(input)
        print(output)

def test_nn():
    nn = NNSimple(3, 4, 2)

    input = np.ones((3, 1))
    desired = np.ones((2, 1))
    output = nn.predict(input)
    print(output)

    print_every_n = 25
    #try again and again
    for i in range(0, 1000):
        nn.learn(desired)
        output = nn.predict(input)
        if i % print_every_n == 0:
            print(output)

test_layer()
# test_nn()