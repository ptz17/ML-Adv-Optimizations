### Homework 7
# Accelerated Gradient Descent implementation with least mean squares.
#
# Princess Tara Zamani
# 10/30/2021

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
def Accelerated_GradientDescent(x0, y0, A, b, z, k_iterations, momentum_coeff_method, lr):
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
        inner_calc = np.matmul(A, y_current) - b

        # Accumulate data for plotting
        y_pred[k] = 1/2 * np.square(np.linalg.norm(inner_calc, 2)) # Accumulate predicted f(x) 

        # Update Momentum coefficient if necessary
        if momentum_coeff_method == "(1) Iteration Based":
            momentum_coeff = k / (k+3)


        grad = np.matmul(A.T, inner_calc)  # Calculate gradient f(y_k)
        x_next = y_current - lr*grad    # Update gradient  x_k+1
        y_next = x_next + momentum_coeff*(x_next - x_current) # Update y_k+1
        x_current = x_next  # Update x_current
        y_current = y_next  # Update y_current

    return y_pred

    

def main():
    n = 100 # Dimensions
    m = int(0.1*n)  # Number of measurements
    var = 0.1   # Noise variance. The bigger = more noise.
    A, b, z = DataGeneration(n, m, var)

    x0 = np.abs(np.random.randn(n,1))  
    y0 = x0.copy()
    lr = 1e-3  # Learning rate
    momentum_coeff_methods= ["(1) Iteration Based", "(2) Proper Constant", "(3) Zero"]
    k = 6000
    
    # Plot training loss for each momentum coeff
    plt.figure(1)
    for momentum_method in momentum_coeff_methods:
        y = Accelerated_GradientDescent(x0, y0, A, b, z, k, momentum_method, lr)
        plt.plot(range(k), y, label=momentum_method)

    plt.title('Training Loss Curve for Accelerated Gradient Descent on LMS')
    plt.xlabel('Iteration Count, $k$')
    plt.ylabel('Loss, $f (y_k)$')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()

