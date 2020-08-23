# set some inputs
x = -2
y = 5
z = -4

# perform the forward pass
q = x + y  # q becomes 3
f = q * z  # f becomes -12

dqdx = 1
dqdy = 1

dfdq = z
dfdz = q

dfdx = 1 * z
dfdy = 1 * z

import numpy as np

a = np.array([
    [1, 4, 2],
    [2, 4, 6],
    [1, 7, 8],
    [2, 5, 7]])

b = np.array([
    [1, 4, 2, 5, 6],
    [2, 4, 6, 2, 1],
    [1, 7, 8, 1, 3]])

print(b.T.shape, a.T.shape)
print(np.matmul(b.T, a.T).shape)


'''


f(x) = 2x^2 + 3
f'(x) = 4x

g(x, y) = 2x^2 + xy + y^3

dgdx = 4x + y
dgdy = x + 3y^2

gradient = [
    dgdx: 4x + y,
    dgdy: x + 3y^2
]

'''

lr = 0.01

x = 5
y = 7
f = 2 * x ** 2 + x * y + y ** 3
print(f)
for i in range(2000):
    g = 2 * x ** 2 + x * y + y ** 3

    print(g)

    dfdx = 4 * x + y
    dfdy = x + 2 * y

    x += -dfdx * lr
    y += -dfdy * lr
