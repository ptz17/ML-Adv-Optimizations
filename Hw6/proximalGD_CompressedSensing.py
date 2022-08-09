### Homework 6
# Proximal Gradient Descent implementation for Compressed Sensing.
#
# Princess Tara Zamani
# 10/22/2021

import numpy as np
import matplotlib.pyplot as plt


def GradientDescent(x0, A, b, z, k_iterations, lr):
    x_current = x0
    est_err = np.zeros(k_iterations)

    # Iteratively update x_k+1
    for i in range(k_iterations):
        inner_calc = np.matmul(A, x_current) - b
        grad = np.matmul(A.T, inner_calc)  # Calculate gradient
        x_next = x_current - lr*grad    # Update gradient 
        x_current = x_next  # Update x_current
        est_err[i] = np.linalg.norm(x_current - z, ord=2) / np.linalg.norm(z, ord=2)  # Accumulate error

    return est_err


def ProximalGradientDescent(x0, A, b, z, k_iterations, lr, lam):
    x_current = x0
    est_err = np.zeros(k_iterations)

    # Iteratively update x_k+1
    for i in range(k_iterations):
        inner_calc = np.matmul(A, x_current) - b
        grad = np.matmul(A.T, inner_calc)  # Calculate gradient
        prox = np.sign(grad) * np.maximum(np.abs(grad) - lam, 0) # proximal operator of lam*||.||_1
        x_next = prox # Update gradient 
        x_current = x_next  # Update x_current
        est_err[i] = np.linalg.norm((x_current - z), ord=2) / np.linalg.norm(z, ord=2)  # Accumulate error

    return est_err


def main():

    # Data Generation
    n = 1000 # Dimensions
    m_list = [100, 400, 700]  # Number of measurements
    variance = 0.1   # Noise variance. The bigger = more noise.

    ###### Gradient Descent for LMS
    f1 = plt.figure(1)
    # Loop through m values
    for m in m_list:
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
        lr = 1e-3   # Learning rate
        k = 7000
        est_error = GradientDescent(x0, A, b, z, k, lr)

        # Plot training loss & error curve
        label = "m = " + str(m)
        print(label)
        plt.plot(range(k), est_error, label=label)
        
    plt.legend()
    plt.title('Gradient Descent for LMS')
    plt.xlabel('Iteration Count')
    plt.ylabel('Estimation Error')
    plt.show()

    #### #Proximal GD for Compressed Sensing
    
    ## Fine tune lambda 
    # m = 100
    # z = np.abs(np.random.randn(n,1)) # Underlying signal 
    # z[:int(0.95 * len(z))] = 0 # Set first 95% of entries in z to zero to make it sparse
    # np.random.shuffle(z) # Shuffle z for random sparsity

    # a = np.random.normal(0, 1, size=(n, m)) # Gaussian measurement vector ~N(0,1)
    # a = a / np.linalg.norm(a,2)   # Normalize vector

    # delta = np.random.normal(0, variance, size=m)
    # b = np.zeros(m)
    # for i in range(m):
    #     b[i] = np.matmul(a[:,i].T, z) + delta[i]   # Observations

    # A = a.T 

    # lambda_list = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    # est_error_avglist = np.zeros(len(lambda_list))
    # for i in range(len(lambda_list)):
    #     x0 = np.abs(np.random.randn(n,1))  
    #     lr = 1e-3   # Learning rate
    #     k = 1000
    #     est_error = ProximalGradientDescent(x0, A, b, z, k, lr, lambda_list[i])
    #     est_error_avglist[i] = np.average(est_error)
    # print(est_error_avglist)

    ## Loop through m values
    f1 = plt.figure(1)
    lam = 0.1 # Fine Tuned value
    for m in m_list:
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
        lr = 1e-3  # Learning rate
        k = 50
        est_error = ProximalGradientDescent(x0, A, b, z, k, lr, lam)

        # Plot training loss & error curve
        label = "m = " + str(m)
        print(label)
        plt.plot(range(k), est_error, label=label)
        
    plt.legend()
    plt.title('Proximal Gradient Descent for Compressed Sensing')
    plt.xlabel('Iteration Count')
    plt.ylabel('Estimation Error')
    plt.show()


if __name__ == "__main__":
    main()