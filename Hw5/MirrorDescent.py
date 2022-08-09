### Homework 5
# Mirror Descent Implementation for solving simplex-constrained Least Mean Squares problem.
#
# Princess Tara Zamani
# 10/16/2021

import numpy as np
import matplotlib.pyplot as plt

### Algorithm Implementation
def GradientDescent(x0, A, b, z, k_iterations, lr):

    x_current = x0
    y_pred = np.zeros(k_iterations)

    # Iteratively update x_k+1
    for i in range(k_iterations):
        inner_calc = np.matmul(A, x_current) - b

        # Accumulate data for plotting
        y_pred[i] = 1/2 * np.square(np.linalg.norm(inner_calc, 2)) # Accumulate predicted f(x) 

        grad = np.matmul(A.T, inner_calc)  # Calculate gradient
        num = x_current * np.exp(-lr * grad)
        den = np.sum(x_current*np.exp(-lr*x_current))
        x_next = num / den # Update gradient 
        x_current = x_next  # Update x_current

    return y_pred#, est_err

    

def main():
    n = 600 # >= 500
    m = 200 # >= 100

    # Data generation from HW 2
    variance = 0.1 # Noise variance. The bigger = more noise.
    z = np.abs(np.random.randn(n,1)) # Underlying signal - column vector 
    a = np.random.normal(0, 1, size=(n, m)) # Gaussian measurement vector ~N(0,1)
    a = a / np.linalg.norm(a,2)   # Normalize vector

    delta = np.random.normal(0, variance, size=m)
    b = np.zeros(m)
    for i in range(m):
        b[i] = np.matmul(a[:,i].T, z) + delta[i]   # Observations

    A = a.T 

    x0 = np.abs(np.random.randn(n,1))  
    lr = 1e-2 # Learning rate
    k = 4000
    y = GradientDescent(x0, A, b, z, k, lr)

    # Plot training loss & error curve
    f1 = plt.figure(1)
    ax1 = f1.add_subplot(121)
    ax1.plot(range(k), y) # f(x_k) vs. k
    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel("Iteration Count")
    ax1.set_ylabel("Loss")
    plt.show()

    
if __name__ == "__main__":
    main()
