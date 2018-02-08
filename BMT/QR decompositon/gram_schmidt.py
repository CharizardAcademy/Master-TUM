# -*- coding: UTF-8

import numpy as np

def gram_schmidt(A):
    Q = None
    Q = np.zeros(A.shape)
    counter = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, counter):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])
        q = u / np.linalg.norm(u)
        Q[:, counter] = q
        counter += 1
    R = np.dot(Q.T, A)
    return Q, R
