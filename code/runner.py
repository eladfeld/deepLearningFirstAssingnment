from mat_file_util import DataLoader
from neural_network import NeuralNetwork

def main():
    path = '../data/SwissRollData.mat'
    data_loader = DataLoader(path)
    layer_dims = [data_loader.input_dim, 2, 2, data_loader.num_labels]
    nn = NeuralNetwork(layer_dims)