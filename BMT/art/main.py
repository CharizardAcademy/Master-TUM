#################################################################
#
#   main.py
#   main file for the demonstration of the ART algorithm
#   written by: Walter Simson
#               Chair for Computer Aided Medical Procedures
#               & Augmented Reality
#               Technical University of Munich
#               27.10.2017
#   based on the work of Maximilian Baust
#
#################################################################

import warnings

import matplotlib.pyplot as plt
import numpy as np

from art import art
from helper import load_data

# Clean up
plt.close('all')

# Load system of equations
# A is the system matrix, b is the right hand side,
# x is the true solution, i.e. Ax=b
A, b, x = load_data("system.mat")
# Set number of iterations
iterations = 10

# Initialize plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
plt.tight_layout()
ax1.set_title('np.linalg.solve')
ax2.set_title('np.linalg.qr')
ax3.set_title('ART')
ax4.set_title('True Solution')

# Plot true solution
x = x.reshape((60, 60))
ax4.imshow(x, extent=[0, 1, 0, 1], cmap='bone')

# Solve LSE with numpy solver
x_np = np.linalg.solve(A, b)
x_np = x_np.reshape((60, 60))
# Warn like MATLAB
x_cond = np.linalg.cond(A)
warnings.warn("Warning: Matrix is close to singular or badly scaled." +
              " Results may be inaccurate. Condition = {0}.".format(x_cond))

ax1.imshow(x_np, extent=[0, 1, 0, 1], cmap='bone')

# Solve LSE with QR-Algorithm
q, r = np.linalg.qr(A)
y = np.matmul(q.T, b)
x_qr = np.linalg.solve(r, y)
x_qr = x_qr.reshape((60, 60))
ax2.imshow(x_qr, extent=[0, 1, 0, 1], cmap='bone')

# Solve LSE with ART
x_art = art(A, b, iterations)
x_art = x_art.reshape((60, 60))
ax3.imshow(x_art, extent=[0, 1, 0, 1], cmap='bone')

plt.show()
