### Homework 4
# Frank-Wolfe Implementation for solving the leading eigenvector of matrix.
#
# Princess Tara Zamani
# 09/27/2021

import numpy as np

## Data Generation
n = 10
A = np.random.normal(0, 1, size=(n, n))
e = 0.75 # Choose 0.5 < e < 1
Q = np.matmul(A.T, A) + e*np.identity(n)

## Algorithm Implementation
lr = 1.75e-2 # 0 < lr < 1
x0 = np.random.rand(n,1)
k_iterations = 3000
x = x0
for k in range(k_iterations):
    x_next = (1-lr)*x + lr*np.matmul(Q,x)/(np.linalg.norm(np.matmul(Q,x), ord=2))
    x = x_next

py_eig_vals, py_eig_vecs = np.linalg.eig(Q)
py_leading_eig_vec = np.max(py_eig_vals) * x
my_leading_eig_vec = np.matmul(Q, x)
print("Python Leading Eig Vector\n", py_leading_eig_vec)
print("\n Frank-Wolfe Leading Eig Vector\n", my_leading_eig_vec)
print("\nDirect Comparison of Equality\n", np.abs(py_leading_eig_vec - my_leading_eig_vec) < 1e-4) 