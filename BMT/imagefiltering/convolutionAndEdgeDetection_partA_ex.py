# Lecture "Basic Math Tools"
# Supplementary Material
#
# Technische Universität München, Fakultät für Informatik
# Dr. Tobias Lasser, Richard Brosig, Jakob Vogel, Dr. Maximilian Baust, Rüdiger Göbl
# 2013

from scipy import signal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# FUNCTION convolution_sl_v2(filter, boundary = 'trim')
#
# Demonstrates convolution, the parameters define the filter and control
# the way the boundary is handled. Possible values are:
#
# 'trim'        Discards values where the full data set is not available
# 'zero'        Pads with zero values.
def convolution_sl_v2(I, filt, boundary='zero'):

    # choose an appropriate parameter for 'convolve2d'
    param_python = 'same'
    if boundary == 'trim':
        param_python = 'valid'

    # compute the result using matlab's built-in function
    result_python = signal.convolve2d(I, filt, param_python)
    plt.figure()
    plt.imshow(result_python, cmap='gray')
    plt.title('python (' + param_python + ')')

    # compute the result using our function
    result_our = do_convolution(I, filt, boundary)

    plt.figure()
    plt.imshow(result_our, cmap='gray')
    plt.title('ours (' + boundary + ')')

    # compate the results using SSD
    diff = result_python - result_our
    plt.figure()
    plt.imshow(diff)
    plt.title('difference')

    plt.figure()
    plt.imshow(I, cmap='gray')
    plt.title('original')

    print('SSD = ' + str(np.sum(diff**2)))
    print('min(diff) = ' + str(np.min(diff)))
    print('max(diff) = ' + str(np.max(diff)))

def wise_element_sum(img,fil):
    return (img * fil).sum()

def do_convolution(image, filt, boundary):
    filt = np.flipud(np.fliplr(filt))
    print(image.shape)
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    print(image_padded.shape)
    image_padded[1:-1, 1:-1] = image
    output = np.zeros_like(image)
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y,x]=(filt*image_padded[y:y+3,x:x+3]).sum()
    print(output.shape)
    return output

I = np.array(Image.open("tire.tif")) / 255
# try the filters from the assignment
myFilter = 1/9 * np.ones((3,3))
convolution_sl_v2(I, myFilter)
myFilter = 1/16 * np.array([[1,2,1],[2,4,2],[1,2,1]])
convolution_sl_v2(I, myFilter)

plt.show()
