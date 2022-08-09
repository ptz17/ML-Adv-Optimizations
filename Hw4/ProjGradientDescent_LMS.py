### Homework 4
# Projected Gradient Descent Implementation for solving Least Mean Squares problem.
#
# Princess Tara Zamani
# 10/5/2021

import numpy as np
import matplotlib.pyplot as plt

### Data Generation
def DataGeneration(n, m, variance):
    z = np.abs(np.random.randn(n,1)) # Underlying signal - column vector 

    a = np.random.normal(0, 1, size=(n, m)) # Gaussian measurement vector ~N(0,1)
    a = a / np.linalg.norm(a,2)   # Normalize vector

    delta = np.random.normal(0, variance, size=m)
    b = np.zeros(m)
    for i in range(m):
        b[i] = np.matmul(a[:,i].T, z) + delta[i]   # Observations

    A = a.T 
    return A, b, z

### Algorithm Implementation
def GradientDescent(x0, A, b, z, k_iterations, lr):
    x_current = x0
    y_pred = np.zeros(k_iterations)
    fits_constraint = np.zeros(k_iterations)

    # Iteratively update x_k+1
    for i in range(k_iterations):
        inner_calc = np.matmul(A, x_current) - b

        # Accumulate data for plotting
        y_pred[i] = 1/2 * np.square(np.linalg.norm(inner_calc, 2)) # Accumulate predicted f(x) 
        grad = np.matmul(A.T, inner_calc)  # Calculate gradient
        x_next = proj(x_current - lr*grad, x_current)   # Update gradient \

        # Accumulate constraint checks 
        if (np.linalg.norm(x_next, np.inf) <= 1):
            fits_constraint[i] = 1 # if in constraint = 1 = True
        else:
            fits_constraint[i] = 0 # Elese = 0 = False
        x_current = x_next  # Update x_current

    return y_pred, fits_constraint

def proj(z, x):
    constraint = np.copy(x)
    constraint[np.linalg.norm(x, ord=1, axis=1) > 1] = np.inf
    dist = np.power(constraint - z, 2)
    idx = np.argmin(dist, axis=1)
    sum = x[idx]
    return sum

def main():
    n = 100 # Dimensions
    m = int(0.1*n)  # Number of measurements
    var = 0.1   # Noise variance. The bigger = more noise.
    A, b, z = DataGeneration(n, m, var)

    x0 = np.abs(np.random.randn(n,1))  
    lr = 1e-3  # Learning rate
    k = 1000
    y, fits_constraint = GradientDescent(x0, A, b, z, k, lr)

    # Constraint check
    print("All x_k in constraint? : ", not fits_constraint.any() == 0) # Checks for any 0s

    # Plot training loss & error curve
    f1 = plt.figure()
    ax1 = f1.add_subplot(121)
    ax1.plot(range(k), y) # f(x_k) vs. k
    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel("Iteration Count")
    ax1.set_ylabel("Loss")
    
    plt.show()


if __name__ == "__main__":
    main()