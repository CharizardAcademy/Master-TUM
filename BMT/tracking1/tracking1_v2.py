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
    residual = np.sqrt(np.sum((np.matmul(a, x) - b)**2))

    return x, residual


# load video and set up coordinate system
# mat_contents = sio.loadmat(tracking-cars.mat')
mat_contents = sio.loadmat('tracking-cars.mat')

# convert movie to double
movie = np.asarray(mat_contents['movie'], np.double)

# get first image (movie is a 3-d array, where the 3rd dimension is the frame number)
# 注意第三个维度是帧数
Is = movie[:,:,0]

# generate coordinate system
# 图像坐标原点在图像中心
# 要用那个给的函数scipy.ndimage.map_coordinates把图像插值插成方阵
[r, c] = Is.shape

#x,y = np.mgrid[:2:240j,:2:240j]

#z = ndimage.map_coordinates(Is,[x.flatten(),y.flatten()],order=1)
#z = z.reshape(240,-1)
# display image
plt.imshow(Is, cmap='gray')

# get center point for patch P
tr = plt.ginput(1) # ginput获取的是像素坐标的一个tuple
# 获取patch中心点的位置
tr_x = [tr[0][0]]
tr_y = [tr[0][1]]

# for testing you can use these coordinates
#tr_x = [200-1]
#tr_y = [144-1]
x = tr_x[0] #随便指定了x和y的位置作为patch_coordinate
y = tr_y[0]
# track the patch
# define 'radius' size of patch
# patch半径
t = 10

# set up patch coordinate system
# patch坐标系在patch的左上角？
[tx, ty] = 2*t+1,2*t+1

# display image again
plt.imshow(Is, cmap='gray')
# draw trajectory
plt.plot(tr_x, tr_y, 'r.')

# tell pyplot that we want an asynchronous figure
plt.ion()
plt.show()


# loop over all images
for i in range(1, movie.shape[2]):
    # get target image 从第二帧开始都是target image
    It = movie[:,:,i]


    # compute function and derivatives
    # 移动一个patch相当于这个patch内所有的pixel进行同等程度的移动，所以遍历整个patch内的所有像素
    patch_coordinate = []
    for patch_x in range(tx):
        for patch_y in range(ty):
            # 这里是减是因为白细胞就是朝着左边跑的,如果改成加的话patch会向右跑
            patch_coordinate.append([patch_x-t,patch_y-t])

    grid = np.array(patch_coordinate) + (y,x)
    # 对image线性插值，生成patch，把用相对位置表示的element插值成像素值
    Is_patch = ndimage.map_coordinates(Is,grid.T)
    It_patch = ndimage.map_coordinates(It,grid.T)
    #计算梯度前把21x21的行向量patch重新reshape成方阵
    gradient = np.gradient(np.reshape(Is_p,(2*t+1,2*t+1)))

    # set up equation system
    # 梯度展成一维，连接后再转置
    A = np.stack([np.ravel(gradient[0]),np.ravel(gradient[1])],axis=0).T
    b = np.ravel(It_patch-Is_patch)

    # solve equation system using Golub's algorithm
    # v更新patch
    [v, res] = golub(A, b)

    # x是列方向，y是行方向，
    x,y = x-v[1],y-v[0]
    # update trajectory
    # 每一次画出所有trajecory上的点

    tr_x.append(x)
    tr_y.append(y)

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
