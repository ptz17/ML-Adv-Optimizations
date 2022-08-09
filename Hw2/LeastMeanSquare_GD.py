
### Homework 2
# Gradient Descent implementation with least mean squares.
#
# Princess Tara Zamani
# 09/13/2021

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
    est_err = np.zeros(k_iterations)

    # Iteratively update x_k+1
    for i in range(k_iterations):
        inner_calc = np.matmul(A, x_current) - b

        # Accumulate data for plotting
        y_pred[i] = 1/2 * np.square(np.linalg.norm(inner_calc, 2)) # Accumulate predicted f(x) 
        est_err[i] = np.linalg.norm(x_current - z) / np.linalg.norm(z)  # Accumulate error

        grad = np.matmul(A.T, inner_calc)  # Calculate gradient
        x_next = x_current - lr*grad    # Update gradient 
        x_current = x_next  # Update x_current

    return y_pred, est_err

    

def main():
    n = 100 # Dimensions
    m = int(0.1*n)  # Number of measurements
    var = 0.1   # Noise variance. The bigger = more noise.
    A, b, z = DataGeneration(n, m, var)

    x0 = np.abs(np.random.randn(n,1))  
    lr = 1e-3  # Learning rate
    k = 6000
    y, est_error = GradientDescent(x0, A, b, z, k, lr)

    # Plot training loss & error curve
    f1 = plt.figure(1)
    ax1 = f1.add_subplot(121)
    ax1.plot(range(k), y) # f(x_k) vs. k
    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel("Iteration Count")
    ax1.set_ylabel("Loss")
    ax2 = f1.add_subplot(122)
    ax2.plot(range(k-1), est_error[1:])
    ax2.set_title('Estimation Error Curve')
    ax2.set_xlabel("Iteration Count")
    ax2.set_ylabel("Error")

    # ##############################################
    # # Impact of measurement-dimension ratio (m/n).
    # # n is fixed. m is adjusted.
    # n = 100
    # m = np.arange(0.1*n, 3*n, 0.5*n) 
    # lr = 1e-3   # Learning rate
    # k = 4000
    # error_arr = []

    # for i in range(len(m)):
    #     A, b, z = DataGeneration(n, int(m[i]), var)
    #     x0 = np.random.randn(n,1)  
    #     y, est_error = GradientDescent(x0, A, b, z, k, lr)
    #     error_arr.append(est_error)

    # # Plot results
    # f2 = plt.figure(2)
    # ax1 = f2.add_subplot(111)
    # ax1.set_title("Impacts of Measurement-Dimension Ratio on Error")
    # ax1.set_xlabel("Iteration Count")
    # ax1.set_ylabel("Error")
    # for i in range(len(error_arr)):
    #     label = "m/n = " + str(m[i]) + "/" + str(n)
    #     ax1.plot(range(len(error_arr[i])), error_arr[i], label=label)
    # ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")


    # ##############################################
    # # Impact of noise-variance
    # n = 100 # Dimensions
    # m = int(0.1*n)  # Number of measurements
    # var_arr = np.arange(0, 2.25, 0.25) # Noise variance. The bigger = the more noise.
    # error_arr = []
    # for i in range(9):
    #     A, b, z = DataGeneration(n, m, var_arr[i])

    #     x0 = np.abs(np.random.randn(n,1)) 
    #     lr = 1e-3   # Learning rate
    #     k = 5000
    #     y, est_error = GradientDescent(x0, A, b, z, k, lr)
    #     error_arr.append(est_error)

    # # Plot results
    # f3= plt.figure(3)
    # ax1 = f3.add_subplot(111)
    # ax1.set_title("Impacts of Noise Variance on Error")
    # ax1.set_xlabel("Iteration Count")
    # ax1.set_ylabel("Error")
    # for i in range(len(error_arr)):
    #     label = "v = " + str(round(var_arr[i], ndigits=2))
    #     ax1.plot(range(len(error_arr[i])), error_arr[i], label=label)
    # ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")


    # ##############################################
    # # Change lr
    # var = 0.1 # Noise variance. The bigger = the more noise.
    # k = 5000
    # loss_arr = []
    # lr = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2]
    # x0 = np.abs(np.random.randn(n,1))
    # for i in range(len(lr)):
    #     A, b, z = DataGeneration(n, m, var)
    #     y, est_error = GradientDescent(x0, A, b, z, k, lr[i])
    #     loss_arr.append(y)

    # # Plot results
    # f4 = plt.figure(4)
    # ax1 = f4.add_subplot(111)
    # ax1.set_title("Impacts of Learning Rate on Training Loss Curve")
    # ax1.set_xlabel("Iteration Count")
    # ax1.set_ylabel("Loss")
    # for i in range(len(loss_arr)):
    #     label = "lr = " + str(lr[i])
    #     ax1.plot(range(len(loss_arr[i])), loss_arr[i], label=label)
    # ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    
    plt.show()


if __name__ == "__main__":
    main()

