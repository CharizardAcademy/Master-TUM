import scipy.io as sio


def load_data(file_name):
    mat_contents = sio.loadmat(file_name)
    A = mat_contents['A']
    b = mat_contents['b']
    x = mat_contents['x']
    return A, b, x
