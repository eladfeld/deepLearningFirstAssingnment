from utils import unison_shuffled_copies
from mat_file_util import DataLoader
import numpy as np
import funcs as f
from NN2 import NN
from data_maker import make_swirl_data


def argmax_data(dim=10, num_samples=25000):
    np.random.seed(42)
    inputs = np.random.rand(num_samples, dim)
    labels = np.random.rand(num_samples, dim)

    for i in range(len(labels)):
        c = np.argmax(inputs[i])
        for j in range(len(labels[0])):
            if j == c:
                labels[i][j] = 1
            else:
                labels[i][j] = 0

    return inputs, labels

def xor_data(dim=10, num_samples=25000):
    np.random.seed(42)
    inputs = np.random.randint(2, size=(num_samples, dim))
    labels = np.random.rand(num_samples, 2)

    for i in range(len(labels)):
        c = int(np.sum(inputs[i]) % 2)
        for j in range(2):
            if j == c:
                labels[i][j] = 1
            else:
                labels[i][j] = 0

    return inputs, labels
    

def main():
    path = '../data/PeaksData.mat'

    #path = '../data/SwissRollData.mat'
    # data_loader = DataLoader(path)

    # inputs = data_loader.training_inputs.T
    # labels = data_loader.training_labels.T
    #inputs, labels = unison_shuffled_copies(inputs, labels)
    inputs, labels = make_swirl_data(10000)
    # inputs = inputs[5:6]
    # labels = labels[5:6]

    in_dim, num_labels, num_samples = get_info(inputs, labels)

    layer_dims = [in_dim, 10, 10, num_labels]
    print_info(in_dim, num_labels, num_samples, layer_dims)

    nn = NN(layer_dims=layer_dims, act=f.tanh, act_tag=f.tanh_tag)
    nn.train(inputs, labels, mini_batch_size=1, batch_size=1000, num_epochs=1)


def single_pass():
    path = '../data/PeaksData.mat'

    #path = '../data/SwissRollData.mat'
    # data_loader = DataLoader(path)

    # inputs = data_loader.training_inputs.T
    # labels = data_loader.training_labels.T
    #inputs, labels = unison_shuffled_copies(inputs, labels)
    inputs, labels = make_swirl_data(10)
    inputs = inputs[5:6]
    labels = labels[5:6]

    in_dim, num_labels, num_samples = get_info(inputs, labels)

    layer_dims = [in_dim, 3, num_labels]
    print_info(in_dim, num_labels, num_samples, layer_dims)

    nn = NN(layer_dims=layer_dims, act=f.tanh, act_tag=f.tanh_tag)
    nn.train(inputs, labels, mini_batch_size=1, batch_size=1, num_epochs=1)



def num_params(dims):
    output = 1
    for dim in dims:
        output *= dim
    return output

def print_info(in_dim, num_labels, num_samples, layer_dims):
    print(f"in dim: {in_dim}")
    print(f"num labels: {num_labels}")
    print(f"num samples: {num_samples}")
    print(f"num weights: {num_params(layer_dims)}")

def get_info(inputs, labels):
    in_dim = len(inputs[0])
    num_labels = len(labels[0])
    num_samples = len(inputs)
    return in_dim, num_labels, num_samples

def print_swiss_points(inputs, labels, c):
    for i in range(len(inputs)):
        if labels[i][c] > -0.1:
            input = inputs[i]
            print(f"({input[0]}, {input[1]})")

# single_pass()
main()





