# -*- coding: UTF-8
# Lecture "Basic Math Tools"
# Eigen-Faces
#
# Technical University of Munich
# Walter Simson
# 2017

import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt


def plot_faces(faces, ih, iw, rows, cols, title):
    # Initialize grid of faces
    grid = np.zeros((ih * rows, iw * cols))

    for row in range(rows):
        for col in range(cols):
            grid[row * ih:(row + 1) * ih, col * iw:(col + 1) * iw] = faces[
                row + col * rows, :].reshape((ih, iw))
    plt.imshow(grid, cmap="bone")
    plt.tight_layout()
    plt.title(title)


def normalize_image(image):
    image_min = np.min(image)
    image_range = np.max(image) - image_min
    return (image - image_min) / image_range


# Import face images
def import_faces():

    # Initialize faces list
    faces = []
    # Dimension 0 is file, 1 is row of face,
    for file in glob.glob("/Users/gaoyingqiang/Desktop/大学/Master/BMT/eigenface/face/*.pgm"):
        # Import and flatten images into vectors
        face = np.array(Image.open(file)).flatten()

        # Normalize Faces and add to list
        faces.append(normalize_image(face))

    # Set image dimensions for square matrix
    #print(np.int(np.sqrt(faces[0].shape[0])))
    iw = ih = np.int(np.sqrt(faces[0].shape[0]))
    # Return 2D array of image vectors as well as image dims
    return np.array(faces), ih, iw


def split_faces(faces, training_size):
    # Shuffle and split faces
    np.random.shuffle(faces) #随机打乱图像的顺序并返回
    training_faces = faces[:training_size] # 取打乱后前training_size张图像作为训练集
    test_faces = faces[training_size:] # 取剩下的图像作为测试集
    return training_faces, test_faces

#### Your work starts here!!
def main():
    # Import the faces from a file
    # Function returns a 2D matrix images
    # image of width (iw) and height (ih)
    faces, ih, iw = import_faces() #返回faces是图片，ih是图片高，iw是图片宽

    # Look at the shape of the faces you get.
    # This command will be helpful later!
    # How many images are imported?
    # What are the dimensions of the image?
    print(faces.shape)

    # Split into two sets; training and testing
    training_faces, test_faces = split_faces(faces, 500) # 生成训练集和测试集

    plt.figure(1)
    plot_faces(training_faces, ih, iw, 10, 10, "Input images (normalized)")

    # Part a.)
    # Calculate the mean of the faces intensity values for every pixel
    faces_mean = np.mean(faces, 0)
    # Subtract the mean faces
    training_faces = np.subtract(training_faces, faces_mean)
    # Plot de-meaned faces
    plt.figure(2)
    plot_faces(training_faces, ih, iw, 10, 10, "De-meaned face")

    # Compute the Eivenvectors of the covariance of the Images
    # Hint: np.linalg.eig() and np.cov()
    # Double hint: read the documentation on np.cov()
    #       Which axis does it operate on?
    [w, v] = np.linalg.eig(np.cov(training_faces.T))

    # Plot your eigen-faces
    # Hint: are your eigen-faces side-ways?
    plt.figure(3)
    plot_faces(v.T, ih, iw, 10, 10, "eigen-faces")

    # Create all linear basis coefficients for test_face images
    test_eig_vec, coefs = np.linalg.eig(test_faces)
    mean_face_error = []
    for compression in range(0, ih * iw):
        reconstructed_faces = np.matmul(coefs[:, :compression],
                                        v.T[:compression, :])
        # Compute the absolute value of the error term (element-wise)
        face_error = np.abs(reconstructed_faces - test_faces)
        # Compute the mean value of the error. Append it to the list.
        mean_face_error.append(np.mean(face_error))

    plt.figure(4)
    plt.semilogy(mean_face_error)
    plt.xlabel("Number of eigenfaces used for compressed image")
    plt.ylabel("Sum absolute error of compressed image")
    plt.title("Sum absolute error")
    plt.figure(5)
    plt.loglog(np.array(mean_face_error[:-1]) - np.array(mean_face_error[1:]))
    plt.xlabel("Number of eigenfaces used for compressed image")
    plt.ylabel(
        "Relative Sum absolute error of compressed image (from previous image)"
    )
    plt.title("Relative error between current and previous approximation ")
    plt.figure(6)
    plt.xlabel("Eigen value number")
    plt.ylabel("Eigen-Value")
    plt.title("Size of eigenvalue")
    plt.semilogy(w)
    plt.tight_layout()
    plt.title("mean_face_error")

    plt.show()


if __name__ == "__main__":
    main()
