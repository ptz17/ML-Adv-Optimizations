
### Homework 9
# Stochastic Gradient Descent implementation for least mean squares with varying sampling methods.
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
def Stochastic_GradientDescent_withReplacement(x0, y0, A, b, z, k_iterations, lr, batchSize, sampling='normal'):
    x_current = x0
    y_current = y0
    y_pred = np.zeros(k_iterations)
    est_err = np.zeros(k_iterations)
    momentum_coeff = 0

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


### Algorithm Implementation
def Stochastic_GradientDescent_withoutReplacement(x0, y0, A, b, z, k_iterations, lr, batchSize, sampling='normal'):
    x_current = x0
    y_current = y0
    epochNum = 20
    k_iterations = int(len(b) / batchSize)
    y_pred = np.zeros(k_iterations)
    # y_pred_final = [] #np.zeros(epochNum)


    for epoch in range(epochNum):
        # Reshuffle data
        y_pred_final = []
        A_reshuffle = np.copy(A)
        np.random.shuffle(A_reshuffle)
        b_reshuffle = np.copy(b)
        np.random.shuffle(b_reshuffle)

        # Iteratively update x_k+1
        for k in range(k_iterations):

            # Randomly sample B data points 
            sampleIdx = list(range(k*batchSize, k+batchSize+batchSize-1)) # without replacement
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

        y_pred_final = np.concatenate((y_pred_final, y_pred))#[epoch] = np.mean(y_pred)
    return y_pred #_final #, est_err
    


def GradientDescent_with_momentum(x0, y0, A, b, z, k_iterations, lr):
    x_current = x0
    y_current = y0
    y_pred = np.zeros(k_iterations)

    momentum_coeff = 0 # k/(k+3) when k = 0 for Iteration Based method initialization

    # Iteratively update x_k+1
    for k in range(k_iterations):
        inner_calc = np.matmul(A, y_current) - b

        # Accumulate data for plotting
        y_pred[k] = 1/2 * np.square(np.linalg.norm(inner_calc, 2)) # Accumulate predicted f(x) 

        # Update Momentum coefficient if necessary
        momentum_coeff = k / (k+3)
        grad = np.matmul(A.T, inner_calc)  # Calculate gradient f(y_k)
        x_next = y_current - lr*grad    # Update gradient  x_k+1
        y_next = x_next + momentum_coeff*(x_next - x_current) # Update y_k+1
        x_current = x_next  # Update x_current
        y_current = y_next  # Update y_current

    return y_pred


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
    samplingTypes = ['normal', 'with_averaging', 'with_momentum']
    # Plot training loss & error curve
    f1 = plt.figure()
    ax1 = f1.add_subplot(131)

    for samplingType in samplingTypes:
        y = Stochastic_GradientDescent_withReplacement(x0, y0, A, b, z, k, lr, batchSize, sampling=samplingType)
        ax1.plot(np.multiply(batchSize, list(range(k))), y, label=samplingType) # f(x_k) vs. k
    ax1.set_title('SGD + Sampling With Replacement')
    ax1.set_xlabel("Number of Data Points")
    ax1.set_ylabel("Loss")
    ax1.legend()


    # Plot training loss & error curve
    # f2 = plt.figure()
    ax2 = f1.add_subplot(132)
    for samplingType in samplingTypes:
        y_withoutReplacement = Stochastic_GradientDescent_withoutReplacement(x0, y0, A, b, z, k, lr, batchSize, sampling=samplingType)
        ax2.plot(np.multiply(batchSize, list(range(5))), y_withoutReplacement, label=samplingType) # f(x_k) vs. k
    ax2.set_title('SGD + Sampling Without Replacement')
    ax2.set_xlabel(f"Number of Data Points")
    ax2.set_ylabel("Loss")
    ax2.legend()

    # f3 = plt.figure()
    ax3 = f1.add_subplot(133)
    y_GD = GradientDescent_with_momentum(x0, y0, A, b, z, k, lr)
    ax3.plot(np.multiply(m, list(range(k))), y_GD) # f(x_k) vs. k
    ax3.set_title('Loss Curve for GD + Momentum')
    ax3.set_xlabel(f"Number of Data Points")
    ax3.set_ylabel("Loss")

    plt.show()


if __name__ == "__main__":
    main()