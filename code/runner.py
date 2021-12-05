from mat_file_util import DataLoader
import numpy as np
import funcs as f
from NN2 import NN

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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

    

def main():
    # path = '../data/PeaksData.mat'
    path = '../data/SwissRollData.mat'
    data_loader = DataLoader(path)

    inputs = data_loader.training_inputs.T
    labels = data_loader.training_labels.T
    #inputs, labels = argmax_data(dim=10)

    in_dim = len(inputs[0])
    num_labels = len(labels[0])
    num_samples = len(inputs)

    print(f"in dim: {in_dim}")
    print(f"num labels: {num_labels}")
    print(f"num samples: {num_samples}")

    # print(f"input shape: {inputs[0][None, :].shape}")
    # print(f"output shape: {labels[0][None, :].shape}")
    layer_dims = [in_dim, 5, 5, 10, 5, num_labels]
    nn = NN(layer_dims=layer_dims, act=f.tanh, act_tag=f.tanh_tag)

    inputs, labels = unison_shuffled_copies(inputs, labels)
    print(f"in shape: {inputs.shape}, labels shape: {labels.shape}")
    # inputs = np.array(inputs)
    # labels = np.array(labels)
    nn.train(inputs, labels, mini_batch_size=1, batch_size=1)




main()
# my_data()





