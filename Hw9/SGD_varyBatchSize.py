### Homework 9
# Stochastic Gradient Descent implementation for least mean squares with sampling with replacement
# and varying batch sizes.
#
# Princess Tara Zamani
# 12/5/2021

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import sample

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
def Stochastic_GradientDescent_withReplacement(x0, y0, A, b, z, k_iterations, alpha, batchSize, sampling='normal'):
    x_current = x0
    y_current = y0
    y_pred = np.zeros(k_iterations)
    est_err = np.zeros(k_iterations)
    momentum_coeff = 0
    lr = alpha * batchSize

    # Iteratively update x_k+1
    for k in range(k_iterations):

        # Randomly sample B data points 
        sampleIdx = np.random.choice(range(len(b)), batchSize) # with replacement
        A_k = A[sampleIdx]
        b_k = b[sampleIdx] # B_k = {A_k, b_k}
        if sampling=='normal':
            inner_calc = np.matmul(A_k, x_current) - b_k
            # Accumulate data for plotting
            y_pred[k] = np.square(np.linalg.norm(np.matmul(A, x_current) - b, 2)) / (2*len(b)) # Accumulate predicted f(x) 
        elif sampling=='with_averaging':
            inner_calc = np.matmul(A_k, x_current) - b_k
            # Accumulate data for plotting
            y_pred_k = np.square(np.linalg.norm(np.matmul(A, x_current) - b, 2)) / (2*len(b)) # Accumulate predicted f(x) 
            y_pred[k] = (np.sum(y_pred[:k]) + y_pred_k) / (k+1)
        elif sampling=='with_momentum':
            inner_calc = np.matmul(A_k, y_current) - b_k
            # Accumulate data for plotting
            y_pred[k] = np.square(np.linalg.norm(np.matmul(A, y_current) - b, 2)) / (2*len(b)) # Accumulate predicted f(x) 


        grad = np.sum(np.matmul(A_k.T, inner_calc)) / batchSize # Calculate stochastic gradient

        if sampling=='with_momentum':
            momentum_coeff = k / (k+3)
            x_next = y_current - lr*grad    # Update gradient  x_k+1
            y_next = x_next + momentum_coeff*(x_next - x_current) # Update y_k+1

            x_current = x_next  # Update x_current
            y_current = y_next  # Update y_current
        else: 
            x_next = x_current - lr*grad    # Update gradient 
            x_current = x_next  # Update x_current

    return y_pred#, est_err


def main():
    n = 1000 # Dimensions
    m = int(0.1*n)  # Number of measurements
    var = 0.1   # Noise variance. The bigger = more noise.
    A, b, z = DataGeneration(n, m, var)
    print(A.shape, b.shape)
    x0 = np.abs(np.random.randn(n,1))  
    y0 = x0.copy()
    lr = 1e-3  # Learning rate
    batchSize = 20 # Batch size
    k = 100
    # Plot training loss & error curve
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    batchSizes = [1, 10, 20, 30, 40, 50, 100]
    for BS in batchSizes:
        y = Stochastic_GradientDescent_withReplacement(x0, y0, A, b, z, k, alpha=0.0001, batchSize=BS, sampling='normal')
        ax1.plot(np.multiply(batchSize, list(range(k))), y, label=f"Batch Size = {BS}") # f(x_k) vs. k
    ax1.set_title('SGD + Sampling With Replacement')
    ax1.set_xlabel("Number of Data Points")
    ax1.set_ylabel("Loss")
    ax1.legend()

    plt.show()


if __name__ == "__main__":
    main()