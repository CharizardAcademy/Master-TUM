# Lecture "Basic Math Tools"
# Supplementary Material
#
# Technische Universit�t M�nchen, Fakult�t f�r Informatik
# Dr. Tobias Lasser, Dr. Maximilian Baust, Richard Brosig,
# Jakob Vogel, Oliver Zettinig, Nicola Rieke, Anca Stefanoiu, R�diger G�bl
# 2009-2017
#
# This code has originally been written by Maximilian Baust and has been
# adapted to match into the structure of the exercises. The original
# file comment follows below.
#
# This code of for educational purposes only and is based on the
# ideas of Lucas and Kanade
# (Lucas B D and Kanade T 1981, An iterative image registration technique
# with an application to stereo vision. Proceedings of Imaging
# understanding workshop, pp 121--130)
#
# The video is taken from Dirk-Jan Kroon, Lucas Kanade affine template
# tracking, MATLAB central, File Exchange
# Copyright (c) 2009, Dirk-Jan Kroon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
#

import scipy.io as sio
import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt


def golub(a, b):
    # QR decomposition of A
    [q, r] = np.linalg.qr(a)

    # compute new b
    bnew = np.matmul(q.T, b)

    # get size of A
    [m, n] = a.shape
    # compute solution
    x = np.linalg.solve(r[0:n, :], bnew[0:n])

    # compute residual
    residual = np.sqrt(np.sum((np.matmul(a, x) - b) ** 2))

    return x, residual


# load video and set up coordinate system
mat_contents = sio.loadmat('tracking-leukocyte.mat')
#mat_contents = sio.loadmat('tracking-cars.mat')

# convert movie to double
movie = np.asarray(mat_contents['movie'], np.double)

# get first image (movie is a 3-d array, where the 3rd dimension is the frame number)
Is = movie[:,:,0]

# generate coordinate system
[r, c] = Is.shape

# display image
plt.imshow(Is, cmap='gray')

# get center point for patch P
#tr_x = [tr[0][0]]
#tr_y = [tr[0][1]]
# for testing you can use these coordinates
tr_x = [200-1]  #200-1
tr_y = [144-1]  #144-1
x = tr_x[0] #随便指定了x和y的位置作为patch_coordinate
y = tr_y[0]

# track the patch
# define 'radius' size of patch
t = 10
edge = 2*t+1
elements_num = (2*t + 1)*(2*t + 1)
# set up patch coordinate system
#[tx, ty] = [tr_x[0] - t, tr_y[0] - t]

# display image again
plt.imshow(Is, cmap='gray')

# draw trajectory
plt.plot(tr_x, tr_y, 'r.')

# tell pyplot that we want an asynchronous figure
plt.ion()

# loop over all images
for i in range(1, movie.shape[2]):
    # get target image
    It = movie[:, :, i]

    # compute function and derivatives
    x1 =int(tr_x[i - 1] - t)
    x11 =int(tr_x[i - 1] + t + 1)
    y1 = int(tr_y[i - 1] - t )
    y11 = int(tr_y[i - 1] + t + 1)
    #print(tr_x,tr_y)
    I1 = It[y1:y11,x1:x11]
    I0 = Is[y1:y11,x1:x11]
    #I0_gradient = np.gradient(I0)

    gradx=np.zeros((2*t+1,2*t+1))
    grady=np.zeros((2*t+1,2*t+1))
    for j in range(2*t+1):
        gradx[:,j]=Is[y1:y11,x1+j]-Is[y1:y11,x1+j+1]
        grady[j,:]=Is[y1+j,x1:x11]-Is[y1+j+1,x1:x11]
        #print(gradx.shape)

    # set up equation system
    a1 = np.array(gradx).reshape((elements_num, 1))
    a2 = np.array(grady).reshape((elements_num, 1))
    A = np.concatenate((a1,a2),axis=1)

   # print(A.shape)
    b = np.array(I1).reshape((elements_num, 1)) - np.array(I0).reshape(elements_num, 1)
    #print(A)
    #print(b)

    # solve equation system using Golub's algorithm
    [v, res] = golub(A, b)
    #print(v)
    #v=np.linalg.solve(np.dot(A.T,A),np.dot(A.T,b))
    #[v,res,r,s]=np.linalg.lstsq(A,b)


    # update trajectory
    tr_x.append(tr_x[i-1]+v[0])
    tr_y.append(tr_y[i-1]+v[1])

    # update the figure
    plt.clf()
    plt.imshow(It, cmap='gray')

    plt.plot(tr_x, tr_y, 'r.-')
    plt.title('frame ' + str(i) + ', residual ' + str(res))

    # give pyplot time to draw
    plt.pause(0.05)

    # update Is
    Is = It

while True:
    plt.pause(0.1)
