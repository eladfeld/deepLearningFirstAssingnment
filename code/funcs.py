import numpy as np

#tested
def softmax(xTw):
    e_x = np.exp(xTw - np.max(xTw))
    output = e_x / np.sum(e_x, axis=1, keepdims=True)
    return output


#tested
def cross_entropy(pred, real):
    (n, m) = pred.shape #TODO: handle edge cases
    return (np.sum(real * np.log(pred), axis=1) / (-m))[0]

# checked, not tested. numbers make sense though. may want to divide by m?
#note: not the gradient, but its negative
def ce_grad_wrt_sm_input(pred, real):
    return (real - pred) / pred.shape[1]
    #return -np.sum(real / pred, axis=0) #wrt sm output

def grad_softmax(real):
    return lambda pred: ce_grad_wrt_sm_input(pred, real)


def grad_softmax_jac(s):
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
    return jac
    # v = jac @ s.T#todo: fix
    # func_print(f"jac as vec: {v}")
    # return v.T
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



#region TESTS
ZERO = 0.00000000001
ONE =  0.9999999999

def test_cross_entropy():
    print("\n\n***\tTEST CE\t***")
    pred = np.array([[ZERO, ZERO, ZERO, ONE]])
    real = np.array([[0,0,0,1]])
    print(f"CE: {cross_entropy(pred, real)}\tshould be: ~0")

    pred = np.array([[ZERO, ZERO, ONE, ZERO]])
    real = np.array([[0,0,0,1]])
    print(f"CE: {cross_entropy(pred, real)}\tshould be: LARGE NUMBER")

    pred = np.array([[.25,.25,.25,.25]])
    real = np.array([[0,0,0,1]])
    print(f"CE: {cross_entropy(pred, real)}\tshould be: 0.3465 (1.386/4)")

    pred = np.array([[.25,.25,.4,.1]])
    real = np.array([[0,0,0,1]])
    print(f"CE: {cross_entropy(pred, real)}\tshould be: 0.57575 (2.303/4)")


def test_cross_entropy_grad():
    print("\n\n***\tTEST CE_Grad\t***")
    pred = np.array([[ZERO, ZERO, ZERO, ONE]])
    real = np.array([[0,0,0,1]])
    print(f"CEG: {ce_grad_wrt_sm_input(pred, real)}")

    pred = np.array([[ZERO, ZERO, ONE, ZERO]])
    real = np.array([[0,0,0,1]])
    print(f"CEG: {ce_grad_wrt_sm_input(pred, real)}")

    pred = np.array([[.25,.25,.25,.25]])
    real = np.array([[0,0,0,1]])
    print(f"CEG: {ce_grad_wrt_sm_input(pred, real)}")

    pred = np.array([[.25,.25,.4,.1]])
    real = np.array([[0,0,0,1]])
    print(f"CEG: {ce_grad_wrt_sm_input(pred, real)}")


#endregion
# test_cross_entropy()
# test_cross_entropy_grad()

