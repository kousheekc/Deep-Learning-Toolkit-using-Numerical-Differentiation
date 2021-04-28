import numpy as np

def Sigmoid(x):
    return 1/(1 + np.exp(-x))

def Softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()