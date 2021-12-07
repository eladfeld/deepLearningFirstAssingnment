import numpy as np

PI = np.pi
        

def make_swirl_data(n):
    inputs = []
    labels = []
    for t in np.arange(0,  PI*4, PI*4 / n):
        x = t * np.cos(t) / 10
        y = t * np.sin(t) / 10
        inputs.append([x, y])
        labels.append([1.0, 0.0])


    for t in np.arange(0,  PI*4, PI*4 / n):
        x = t * np.cos(t + PI) / 10
        y = t * np.sin(t + PI) / 10
        inputs.append([x, y])
        labels.append([0.0, 1.0])

    return np.array(inputs), np.array(labels)


def test_swirl_maker():
    inputs, labels = make_swirl_data(500)
    for i in range(len(inputs)):
        input = inputs[i]
        print(f"({input[0]}, {input[1]})")
