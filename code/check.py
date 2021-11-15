import numpy as np

val = 1

# A = np.matrix([\
#     [1,2],\
#     [4,5]\
# ])

# # print(A)
# x = np.matrix([0,1,2])

A_trans = np.random.random((2,3))
x_trans = np.random.random((1,2))
print('----- AT -----')
print(A_trans)
print('----- xT -----')
print(x_trans)
print('----- A * x -----')
# C = x * A
C = np.dot(x_trans,A_trans)

print(C)