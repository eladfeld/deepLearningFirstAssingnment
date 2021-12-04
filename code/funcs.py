import numpy as np

def softmax(xTw):
    #assumes x = xTranspose * w in the input
    e_x = np.exp(xTw - np.max(xTw))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
    # return e_x / e_x.sum()

    # x_transpose = x.T
    # n = np.argmax(x_transpose.dot(w))

    # denominator = np.dot(x_transpose, w) - n
    # denominator = np.exp(denominator)
    # denominator = denominator.sum()

    # numerator = np.exp( np.dot(x_transpose, w[index]) - n)

    # return numerator / denominator

def cross_entropy(xTw, one_hot_vector):
    (n, m) = xTw.shape #TODO: handle edge cases
    return (-1/m) * np.sum(np.log(xTw) * one_hot_vector)

def grad_softmax(s):
    sm = softmax(s)
    return sm * (np.ones(sm.shape) - sm)
    # jacobian_m = np.diag(s)

    # for i in range(len(jacobian_m)):
    #     for j in range(len(jacobian_m)):
    #         if i == j:
    #             jacobian_m[i][j] = s[i] * (1-s[i])
    #         else: 
    #             jacobian_m[i][j] = -s[i]*s[j]
    # return jacobian_m
    

def tanh(x):
    return np.tanh(x)

def tanh_tag(x):#todo: need to apply element by element
    #print(f"tan` in: {x}")
    output = 1 - (np.tanh(x) * np.tanh(x))
    #print(f"tan` out: {output}")
    return output

def identity(x):
    return x

def identity_tag(x):
    return np.ones(x.shape)


def get_points(f, a, b, dx):
    output = []
    for x in np.arange(a, b, dx):
        output.append((x, f(x)))
    return output

def print_func(f, a, b, dx, filename):
    with open(file=filename, mode="w") as file:
        for (x, y) in get_points(f, a, b, dx):
            file.write(f"({x},{y})\n")
        file.close()

def print_softmax_points():
    M = 1
    dx = .01
    points = []
    for x in np.arange(0, M, dx):
        x1, x2 = x, M - x
        v = np.array([[x1, x2]])
        y = grad_softmax(v)
        points.append((x1, x2, y))
    with open(file="softmax_tag.txt", mode="w") as file:
        for (x1, x2, y) in points:
            file.write(f"({x1},{x2})\t=>\t{y}\n")
        file.close()
    

print_softmax_points()