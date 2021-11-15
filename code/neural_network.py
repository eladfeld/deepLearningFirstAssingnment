import numpy as np

class NeuralNetwork:

    #TODO: add other params like sigmoid func, learning rate
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.W = self.init_weight_mats(layer_dims)
        self.B = self.init_bias_vectors(layer_dims)#includes output layer. excludes input layer
        self.stored_values = [None] * (len(layer_dims) - 1) #includes output layer. excludes input layer

    def init_weight_mats(self, layer_dims):
        output = []
        for i in range(0, len(layer_dims) - 1):
            # W = np.random.random((layer_dims[i], layer_dims[i+1]))

            #deterministic init until W = ...
            height = layer_dims[i]
            width = layer_dims[i+1]
            mat = []
            for a in range(0, height):
                row = []
                for b in range(0, width):
                    row.append(b)
                mat.append(row)
            W = np.matrix(mat)
            
            output.append(W)
        return output


    def init_weights_vector(self, layer_dims):
        length = 0
        for i in range(0, len(layer_dims) - 1):
            length += (layer_dims[i] * layer_dims[i + 1])
        output = [0.0] * length
        for i in range(0, length):
            output[i] = np.random.uniform(0,1)
        return output

    def init_bias_vectors(self, layer_dims):
        output = []
        for i in range(1, len(layer_dims)): #TODO: check if end_idx should be 1 less (does output layer have biases?)
            dim = layer_dims[i]
            # B = np.random.random((1, dim))
            B = np.zeros((1,dim))#TODO: replace with line above, this is just for deterministic h=behavior
            output.append(B)
        return output


    def get_weights_at_layer(self, layer):
        start_idx = 0
        for i in range(0, layer - 1):
            start_idx += (self.layer_dims[i] * self.layer_dims[i + 1])
        end_idx = start_idx + (self.layer_dims[layer] * self.layer_dims[layer + 1])
        return self.W[start_idx : end_idx]
    
    def get_biases_at_layer(self, layer):
        #TODO: implement
        return []

    def predict(self, input):
        #TODO: throw exception if input bad length
        #TODO: save values of each neuron along the way
        x = input
        for i in range(0, len(self.layer_dims) - 1):
            W = self.W[i]
            B = self.B[i]
            x = np.dot(x, W)
            x = x + B
        return x


def check():
    nn = NeuralNetwork([2,3,4])
    input = np.array([1,2]) #np.matrix([[1,2]])
    print('---INPUT---')
    print(input)
    prediction = nn.predict(input)
    print('---PRED---')
    print(prediction)

check()
