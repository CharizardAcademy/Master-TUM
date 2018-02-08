# -*- coding: UTF-8

# Lecture "Basic Math Tools"
# Supplementary Material
#
# Technische Universität München, Fakultät für Informatik
# Walter Simson
# 2017


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def pause():
    try:
        input("Press Enter to continue...")
    except:
        pass

# Make Matplotlib non-blocking
plt.ion()

# Import the image using Image from pillow (already imported above)
# hint convert it to a numpy array with np.asarray()
im = np.array(Image.open("tire.tif"))
# Use NumPy to calculate the Singular Value Decomposition of the image

R = np.zeros(im.shape)
print(im.shape)
sample_size = 100 #应该是取前200个奇异值合成原图像
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
plt.suptitle('Image compression example')
ax1.set_title('Original image')
ax2.set_title('Reconstructed\nimage R')
ax3.set_title('Image singular\nvector Ri component')
plt.tight_layout()
plt.show()
ax1.imshow(im,cmap='bone')

U,sigma,V_trans = np.linalg.svd(im)

for i in range(sample_size):
    # Visualize some rank 1 matrices (Ri)
    Ri = sigma[i]*np.outer(U[:,i],V_trans[i,:])
    # Approximate the original image (R)
    R = R + Ri
    if i in range(0,sample_size,10):
        plt.suptitle('Image compression example step {}'.format(i))
        ax3.imshow(Ri,cmap='bone')
        ax2.imshow(R,cmap='bone')
        plt.show()
        plt.pause(2)
        pause()
