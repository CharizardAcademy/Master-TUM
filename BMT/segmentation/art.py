# -*- coding: UTF-8
import numpy as np


def art(A, b, iterations):

    # For help with numpy (the numerical programming library for Python) check out this resource:
    # https://www.safaribooksonline.com/library/view/python-for-data/9781449323592/ch04.html
    x = np.zeros((A.shape[1]))
    # Initialize variables squared_norm (see numpy.zeros)
    a = []
    for i in range(0,A.shape[0]):
        a.append(np.linalg.norm(A[i,:],ord=2))
    # Iterate over rows and compute the squared norm row-wise (we will need this in a second)
    # Hint: look into ranges
    for i in range(0,iterations):
    # Iterate over iterations
        for j in range(0,A.shape[0]):
        # Iterate over matrix rows
            x = x + (b[j] - np.dot(A[j,:],x))*A[j,:].T/(a[j]*a[j])
            # x' = x + correction
    return x
