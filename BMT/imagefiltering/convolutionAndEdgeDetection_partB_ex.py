from scipy import signal
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

# number of scales
N = 5

# initial gamma
gamma = 0.5

# load image
I = np.array(Image.open("tire.tif"))/255
row,col = I.shape
# define Gaussian filter
r = 1 # 3*3 Gaussian filter
sumG = 0
G_matrix = np.zeros((2*r+1,2*r+1))
for i in range(0,2*r+1):
    for j in range(0,2*r+1):
        #gaussp = (1/(2*np.pi*((r/3)**2)))* math.e**(-((i-r)**2+(j-r)**2)/(2*((r/3)**2)))
        gaussp = (1/(2*np.pi*(r**2))) * math.e**(-(i-r)**2+(j-r)**2)/(2*(r**2))
        G_matrix[i,j] = gaussp
        sumG = sumG + gaussp
for i in range(0,2*r+1):
    for j in range(0,2*r+1):
        G_matrix[i,j] = G_matrix[i,j]/sumG

#print(G_matrix)
# Note: there are also functions to simply apply a gaussian filter
#   e.g. scipy.ndimage.filters.gaussian_filter
# But here the goal is to explicitly create the filter matrix.
# Can you use scipy.ndimage.filters.gaussian_filter to compute the filter matrix?
#  Hint: Which signal do you need to convolve with a gaussian to have exactly the filter as a result?

# compute gradient magnitude and edge indicator function
# for original scale
scale_space = np.zeros((6,row,col))
gradMag = np.zeros((6,row,col))
g = np.zeros((6,1))
scale_space[0,:,:] = I
for i in range(1,6):
    scale_space[i,:,:] = signal.convolve2d(scale_space[i-1,:,:],G_matrix,mode='same')

for i in range(0,6):
    gradMag[i,:,:] = ndimage.gaussian_gradient_magnitude(scale_space[i,:,:],0.4)
    g[i,:] = 1/(1+(np.linalg.norm(gradMag[i,:,:],ord=2)/gamma**2))

print(g)

fig,axes = plt.subplots(4,3,figsize=(8,6))

axes[0,0].imshow(scale_space[0,:,:],cmap='gray')
axes[0,0].axis('off')

axes[0,1].imshow(scale_space[1,:,:],cmap='gray')
axes[0,1].axis('off')

axes[0,2].imshow(scale_space[2,:,:],cmap='gray')
axes[0,2].axis('off')

axes[1,0].imshow(scale_space[3,:,:],cmap='gray')
axes[1,0].axis('off')

axes[1,1].imshow(scale_space[4,:,:],cmap='gray')
axes[1,1].axis('off')

axes[1,2].imshow(scale_space[5,:,:],cmap='gray')
axes[1,2].axis('off')

axes[2,0].imshow(gradMag[0,:,:],cmap='gray')
axes[2,0].axis('off')

axes[2,1].imshow(gradMag[1,:,:],cmap='gray')
axes[2,1].axis('off')

axes[2,2].imshow(gradMag[2,:,:],cmap='gray')
axes[2,2].axis('off')

axes[3,0].imshow(gradMag[3,:,:],cmap='gray')
axes[3,0].axis('off')

axes[3,1].imshow(gradMag[4,:,:],cmap='gray')
axes[3,1].axis('off')

axes[3,2].imshow(gradMag[5,:,:],cmap='gray')
axes[3,2].axis('off')

plt.show()
