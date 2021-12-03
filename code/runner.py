from mat_file_util import DataLoader
from NN2 import NN

def main():
    path = '../data/PeaksData.mat'
    data_loader = DataLoader(path)

    inputs = data_loader.training_inputs.T
    labels = data_loader.training_labels.T

    in_dim = len(inputs[0])
    num_labels = len(labels[0])
    num_samples = len(inputs)

    print(f"in dim: {in_dim}")
    print(f"num labels: {num_labels}")
    print(f"num samples: {num_samples}")

    # print(f"input shape: {inputs[0][None, :].shape}")
    # print(f"output shape: {labels[0][None, :].shape}")
    print("starting")
    layer_dims = [in_dim, 10, 20, 20, 20, 10, num_labels]
    nn = NN(layer_dims=layer_dims, lr=0.1)
    nn.train(inputs, labels)
    print("complete!")


main()