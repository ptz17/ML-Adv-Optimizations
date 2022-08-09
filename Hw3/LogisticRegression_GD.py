### Homework 3
# Gradient Descent implementation for a logistic regression problem.
#
# Princess Tara Zamani
# 09/21/2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Data Pre-processing
def load_data(fileName):
    # Load data from .csv file
    data = pd.read_csv(fileName)
    # For each feature, subtract by the mean and divided by the std of the entries of this column. Do not do this for labels.
    colNum = len(data.columns)
    data.iloc[:, :colNum-1] -= data.iloc[:, :colNum-1].mean(axis=0)
    data.iloc[:, :colNum-1] /= data.iloc[:, :colNum-1].std(axis=0)
    # Divide the entire 3656 data samples into a training set (first 2656 samples) and a testing set(last 1000 samples)
    train_data = data.iloc[:2656, :]
    test_data = data.iloc[2656:, :]
    return train_data, test_data

def h_wb(x, w, b):
    return 1/(np.exp(-np.matmul(w.T, x) + b))

### Algorithm Implementation
def GradientDescent(x, y, w0, b0, n, k_iterations, lr):
    w = w0
    b = b0
    f_wb = np.zeros(k_iterations)
    est_err = np.zeros(k_iterations)

    for k in range(k_iterations):
        # Calculage gradients
        w_grad = 1/n * np.sum((h_wb(x, w, b) - y)*x, axis=1) 
        b_grad = 1/n * np.sum(h_wb(x, w, b) - y)
        # Update gradients
        # print(w, w_grad)
        w_next = w - lr*w_grad
        b_next = b - lr*b_grad


        # Accumulating data 
        f_wb[k] = -1/n * np.sum((y*np.log(h_wb(x,w,b)) + (1-y)*np.log(1 - h_wb(x,w,b))))
        print(f_wb[k])
        prob_y_is_1 = h_wb(x, w, b)
        prob_y_is_0 = 1 - prob_y_is_1
        predictions = (prob_y_is_1 > prob_y_is_0).astype(int)
        correct = 0
        for i in range(len(y)):
            if (predictions.iloc[i] == y.iloc[i]):
                correct += 1
        rate = correct / len(y)
        est_err[k] = rate

        # Update iterative variables
        w = w_next
        b = b_next
    
    return w, b, f_wb, est_err # Returns w and b obtained after training.
   


def main():
    # Load and pre-process data
    train_data, test_data = load_data('framingham_clean.csv')

    # Training
    x_train = train_data.iloc[:, :15].T
    y_train = train_data.TenYearCHD
    n = len(train_data) 
    d = len(train_data.iloc[0]) - 1 # not including label
    w0 = np.random.rand(d).T # column vector 
    b0 = np.random.rand(1)
    print(w0)
    k_iterations = 100
    lr = 1e-3
    
    learned_w, learned_b, f_wb, error = GradientDescent(x_train, y_train, w0, b0, n, k_iterations, lr)

    print(learned_w, learned_b)
    f1 = plt.figure(1)
    ax1 = f1.add_subplot(121)
    ax1.plot(range(k_iterations), f_wb)
    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    # Testing Predictions 
    # P(y=1 | x,w,b)
    x_test = test_data.iloc[:, :15].T
    y_test = test_data.TenYearCHD
    prob_y_is_1 = h_wb(x_test, learned_w, learned_b)
    prob_y_is_0 = 1 - prob_y_is_1
    predictions = (prob_y_is_1 > prob_y_is_0).astype(int)

    correct = 0
    for i in range(len(y_test)):
        if (predictions.iloc[i] == y_test.iloc[i]):
            correct += 1
    rate = correct / len(y_test)
    print(predictions, y_test)
    print(rate)

    f2 = plt.figure(1)
    ax2 = f2.add_subplot(122)
    ax2.plot(range(len(error)), error)
    ax2.set_title('Testing Error Curve')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Error')

    plt.show()




if __name__ == "__main__":
    main()
