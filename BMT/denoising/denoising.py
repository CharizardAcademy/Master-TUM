# -*- coding: UTF-8
#####################################################################
#
#   denoising.py
#   main file for the demonstration of the Burt Adelson pyramid
#   written by: Maximilian Baust & Rüdiger Göbl
#               Chair for Computer Aided Medical Procedures
#               & Augmented Reality
#               Technische Universität München
#               10-22-2017
#
#####################################################################



import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


# this function performs the linear combinaton of the images.
# you will have to enter your implementation here
#alpha-系数向量
def linear_combination(alpha, d, r):
    i_alpha = np.tile(np.zeros(len(d)),(len(d),1)) #i_alpha是一个64x64的二维数组
    for i in range(0,len(d[0,0,:])):
        i_alpha = i_alpha + alpha[i] * d[:,:,i]
    i_alpha = i_alpha + r
    return i_alpha

def cost_function(alpha, i_orig, d, r):
    # compute linear combination
    i = linear_combination(alpha, d, r)

    # compute sum of absolute differences
    sad = np.sum(np.abs(i_orig - i))
    return sad

# load image data
data = np.load('data.npz')
I_orig = data['I_orig']
I_noisy = data['I_noisy']
R = data['R']
D = data['D']

# show original image
plt.figure()
plt.imshow(I_orig, cmap="gray")
plt.title('original image')
plt.draw()


# show noisy image
plt.figure()
plt.imshow(I_noisy, cmap="gray")
plt.title('noisy image')
plt.draw()

# show difference images and residual part
fig = plt.figure()
fig.add_subplot(2, 3, 1)
plt.imshow(D[:, :, 0], cmap="gray")
plt.title('difference image 1')
plt.draw()

fig.add_subplot(2, 3, 2)
plt.imshow(D[:, :, 1], cmap="gray")
plt.title('difference image 2')
plt.draw()

fig.add_subplot(2, 3, 3)
plt.imshow(D[:, :, 2], cmap="gray")
plt.title('difference image 3')
plt.draw()

fig.add_subplot(2, 3, 4)
plt.imshow(D[:, :, 3], cmap="gray")
plt.title('difference image 4')
plt.draw()

fig.add_subplot(2, 3, 5)
plt.imshow(D[:, :, 4], cmap="gray")
plt.title('difference image 5')
plt.draw()

fig.add_subplot(2, 3, 6)
plt.imshow(R, cmap="gray")
plt.title('residual part')
plt.draw()

# show sum of difference imagesc and residual part
plt.figure()
plt.imshow(np.sum(D, -1) + R, cmap="gray")
plt.title('sum of difference images and residual part')

# find optimal linear combination
alpha0 = np.ones(5)
alpha = scipy.optimize.fmin(
    lambda a : cost_function(a, I_orig, D, R),
    x0=alpha0
)

# show found linear combination
plt.figure()
plt.imshow(linear_combination(alpha, D, R), cmap="gray")
plt.title('denoised image')

# show difference to original image
plt.figure()
plt.imshow(abs(I_orig - linear_combination(alpha, D, R)), cmap="gray")
plt.title('difference to original image')
plt.show()
