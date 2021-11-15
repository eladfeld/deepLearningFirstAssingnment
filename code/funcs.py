import numpy as np

def softmax(xTw):
    #assumes x = xTranspose * w in the input
    e_x = np.exp(xTw - np.max(xTw))
    return e_x / e_x.sum()

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
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m
    
