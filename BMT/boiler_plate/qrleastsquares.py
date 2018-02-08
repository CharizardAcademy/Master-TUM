#################################################################
#
#   qrleastsquares.py
#   written by: Walter Simson
#               Chair for Computer Aided Medical Procedures
#               & Augmented Reality
#               Technical University of Munich
#               27.10.2017
#
#################################################################
import numpy as np


def pause():
    try:
        input('Press Enter to continue...')
    except:
        pass


def namestr(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj][0]


def printVariable(array):
    print("{}: ".format(namestr(array)))
    print(array.astype('f'))


def my_qr(A):
    # Your implementation of QR decomposition via the Gramm-Schmidt
    # method
    #Q = np.zeros(A.shape)
    #R = np.zeros((A.shape[1], A.shape[1]))

    # Iterate through all columns

        # Iterate through rows until current column (the equivalent of upper-triangular elements)

            # Do core Gramm-Schmidt

        # Normalization step
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
    return [Q, R]


# Make nice array output
np.set_printoptions(precision=3)

# Initialize random array
A = np.random.rand(5, 5)

# Test my_qr and numpy version
[Q, R] = np.linalg.qr(A)
[myQ, myR] = my_qr(A)

# Compare the results
printVariable(Q)
printVariable(myQ)
printVariable(R)
printVariable(myR)
pause()

# Caution: this method does not always work!
# Find a matrix B for which QR does not work (see. part a)
B = np.array([[0]])
[myQ, myR] = my_qr(B)
printVariable(myQ)
printVariable(myR)
pause()

# Now let try to find an unsolvable solve an LSE (see. part 3
epsilon = pow(10, -9)  # Machine precision of floating point numbers
A = np.array([[epsilon, 0], [0, epsilon], [1, 1]])
b = np.array([[1], [2], [0]])

# Try to solve with the normal equation A^TAx=A^Tb
try:
    x_ne, residual_ne = np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, b))
except np.linalg.linalg.LinAlgError:
    print("LinAlg Error: Matrix is Singular")
pause()

# Solve the least squares problem using QR
[Q, R] = my_qr(A)
x_qr, residual_qr, rank_qr, s_qr = np.linalg.lstsq(R, np.dot(Q.T, b))
printVariable(residual_qr)
printVariable(x_qr)
