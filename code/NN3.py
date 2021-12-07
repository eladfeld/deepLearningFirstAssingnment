import numpy as np
import funcs as f

LR = 0.001


class Layer:
    def __init__(self, in_dim, out_dim, act, d_act):
        np.random.seed(42)
        self.W = np.random.normal(size=(out_dim, in_dim), loc=0.0, scale=1.0 )
        self.B = np.random.normal(size=(out_dim, 1), loc=0.0, scale=1.0 )
        self.act = act
        self.d_act = d_act

        self.Y = np.zeros((out_dim, 1))
        self.X = np.zeros((in_dim, 1))
        self.df_dtheta = None

    def forward(self, X):
        self.X = X
        Zw = (self.W @ X)
        Zb = Zw + self.B
        output = self.act(Zb)
        self.Y = output


        print(f"X: {X.shape}\tW: {self.W.shape}\t(W*x): {Zw.shape}\tB: {self.B.shape}\t(Wx+b): {Zb.shape}")
        print(f"X:\n {X}\n\nW:\n {self.W}\n\n(W*x):\n {Zw}\n\nB:\n {self.B}\n\n(Wx+b):\n {Zb}")
        return output

class NN:
    def __init__(self, layer_dims, act=f.tanh, d_act=f.tanh_tag, lr=LR):
        self.lr = lr
        self.num_layers = len(layer_dims) - 1

        layers = [None] * self.num_layers

        for i in range(0, self.num_layers - 1):
            layers[i] = Layer(layer_dims[i], layer_dims[i+1], act, d_act)
        layers[self.num_layers - 1] = Layer(layer_dims[0], layer_dims[1], lambda p: f.softmax(p.T), None)#todo provide softmax grad param
        self.layers = layers

    def predict(self, X):
        pred = X
        for i in range(self.num_layers):
            layer = self.layers[i]
            pred = layer.forward(pred)
        return pred

def check():
    np.random.seed(42)
    layer_dims = [2,3]
    nn = NN(layer_dims)
    input = np.array([[0, 1]]).T
    pred = nn.predict(input)
    print(pred)

# check()