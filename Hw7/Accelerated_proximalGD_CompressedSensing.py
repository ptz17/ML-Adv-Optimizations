### Homework 7
# Accelerated Proximal Gradient Descent implementation for Compressed Sensing.
#
# Princess Tara Zamani
# 10/30/2021

import numpy as np
import matplotlib.pyplot as plt


def Accelerated_ProximalGradientDescent(x0, y0, A, b, z, k_iterations, momentum_coeff_method, lr, lam):
    x_current = x0
    y_current = y0
    y_pred = np.zeros(k_iterations)

    # Create momentum coefficient based on method
    if momentum_coeff_method == "(2) Proper Constant":
        momentum_coeff = 0.5
    elif momentum_coeff_method == "(3) Zero":
        momentum_coeff = 0
    else:
        momentum_coeff = 0 # k/(k+3) when k = 0 for Iteration Based method initialization

    # Iteratively update x_k+1
    for k in range(k_iterations):
        # Update Momentum coefficient if necessary
        if momentum_coeff_method == "(1) Iteration Based":
            momentum_coeff = k / (k+3)

        inner_calc = np.matmul(A, y_current) - b
        grad = np.matmul(A.T, inner_calc)  # Calculate gradient
        gd_update_rule = y_current - lr*grad
        prox = np.sign(gd_update_rule) * np.maximum(np.abs(gd_update_rule) - lam, 0) # proximal operator of lam*||.||_1
        x_next = prox # Update gradient 
        y_next = x_next + momentum_coeff*(x_next - x_current)
        x_current = x_next  # Update x_current
        y_current = y_next  # Update y_current

        y_pred[k] = 1/2 * np.square(np.linalg.norm(inner_calc, 2)) + lam*np.linalg.norm(y_current, ord=1) # Accumulate predicted F(y_k) 

    return y_pred #est_err


def main():

    # Data Generation
    n = 1000 # Dimensions
    m = int(0.1*n)  # Number of measurements
    variance = 0.1   # Noise variance. The bigger = more noise.

    #### Accelerated Proximal GD for Compressed Sensing

    lam = 0.1 # Fine Tuned value
    z = np.abs(np.random.randn(n,1)) # Underlying signal 
    z[:int(0.95 * len(z))] = 0 # Set first 95% of entries in z to zero to make it sparse
    np.random.shuffle(z) # Shuffle z for random sparsity

    a = np.random.normal(0, 1, size=(n, m)) # Gaussian measurement vector ~N(0,1)
    a = a / np.linalg.norm(a,2)   # Normalize vector

    delta = np.random.normal(0, variance, size=m)
    b = np.zeros(m)
    for i in range(m):
        b[i] = np.matmul(a[:,i].T, z) + delta[i]   # Observations

    A = a.T 

    # Gradient Descent for LMS 
    x0 = np.abs(np.random.randn(n,1))  
    y0 = x0.copy()
    lr = 1e-3  # Learning rate
    momentum_coeff_methods= ["(1) Iteration Based", "(2) Proper Constant", "(3) Zero"]
    k = 25

    # Plot training loss for each momentum coeff
    plt.figure(1)
    for momentum_method in momentum_coeff_methods:
        y = Accelerated_ProximalGradientDescent(x0, y0, A, b, z, k, momentum_method, lr, lam)
        plt.plot(range(k), y, label=momentum_method)

    plt.title('Training Loss Curve for Accelerated Proximal Gradient Descent on Compressive Sensing')
    plt.xlabel('Iteration Count, $k$')
    plt.ylabel('Loss, $F (y_k)$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()