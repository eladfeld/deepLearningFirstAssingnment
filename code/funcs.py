import numpy as np

def softmax(xTw):
    #assumes x = xTranspose * w in the input
    # e_x = np.exp(xTw - np.max(xTw))
    e_x = np.exp(xTw - np.max(xTw))
    output = e_x / np.sum(e_x, axis=1, keepdims=True)
    return output


def cross_entropy(xTw, one_hot_vector):
    (n, m) = xTw.shape #TODO: handle edge cases
    return (-1/m) * np.sum(np.log(xTw) * one_hot_vector)

def grad_softmax(s):
    func_print("****grad_softmax***")
    func_print(f"s shape: {s.shape}")
    SM = s.reshape((-1,1))
    func_print(f"SM shape: {SM.shape}")
    d = np.diagflat(s)
    func_print(f"d shape: {d.shape}")
    SM_dot_SMT = np.dot(SM, SM.T)
    func_print(f"SM_dot_SMT shape: {SM_dot_SMT.shape}")
    jac =  d - SM_dot_SMT
    func_print(f"jac shape: {jac.shape}")
    func_print(f"jac:\n {jac}")
    #return jac
    v = np.sum(jac, axis=0)
    func_print(f"jac as vec: {v}")
    return v
    # jacobian_m = np.diag(s)

    # for i in range(len(jacobian_m)):
    #     for j in range(len(jacobian_m)):
    #         if i == j:
    #             jacobian_m[i][j] = s[i] * (1-s[i])
    #         else: 
    #             jacobian_m[i][j] = -s[i]*s[j]
    # return jacobian_m
    

DEBUG = False
def func_print(s):
    if DEBUG:
        print(s)

def tanh(x):
    return np.tanh(x)

def tanh_tag(x):#todo: need to apply element by element
    output = 1 - (np.tanh(x) * np.tanh(x))
    # output = x - (np.tanh(x) * np.tanh(x))
    return output

def identity(x):
    return x

def identity_tag(x):
    return np.ones(x.shape)

def relu(x):
    return x * (x > 0)

def relu_tag(x):
    return 1. * (x > 0)


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
    

