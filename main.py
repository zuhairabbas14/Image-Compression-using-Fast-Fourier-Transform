import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def fft(x, inverse):  # performs 1 dimensional fast fourier transform.
    n = len(x)
    if n <= 1:
        return x
    even_part = fft(x[0::2], inverse)
    odd_part = fft(x[1::2], inverse)
    fraction = [np.exp((2j if inverse else -2j) * np.pi * i / n) * odd_part[i] for i in range(n // 2)]
    return [(even_part[i] + fraction[i]) * (1 / 2 if inverse else 1) for i in range(n // 2)] + \
           [(even_part[i] - fraction[i]) * (1 / 2 if inverse else 1) for i in range(n // 2)]


def fft_2d(fft_image, inverse):  # performs 2 dimensional fast fourier transform.
    image = np.zeros(fft_image.shape)
    image = image.astype(complex)
    for col in range(image.shape[1]):
        image[:, col] = fft(fft_image[:, col], inverse)
    for row in range(image.shape[0]):
        image[row, :] = fft(image[row, :], inverse)
    return image


def get_threshold(image, percent):  # returns the index of the threshold value.
    n, m = image.shape
    mag = np.absolute(image)
    mag = mag.flatten()
    mag.sort()
    pos = int(((100-percent) / 100) * (n * m))
    thresh = mag[-pos]
    return thresh


def compress(image, thresh):  # updates the original array according to the threshold.
    n, m = image.shape
    for i in range(n):
        for j in range(m):
            image[i][j] = image[i][j] if np.absolute(image[i][j]) > thresh else 0.0
    return image


def get_files(path):  # returns a list of files in the specified directory.
    my_list = []
    my_list += [each for each in os.listdir(path) if each.endswith('.tif')]
    return my_list


if __name__ == "__main__":

    input_path = './inputs'
    files = get_files(input_path)

    if not os.path.exists('SyedAbbasOutputs'):
        os.makedirs('SyedAbbasOutputs')

    for file in files:
        img = mpimg.imread(input_path + '/' + file)
        img = fft_2d(img, False)
        threshold = get_threshold(img, 95)
        img = compress(img, threshold)
        img = fft_2d(img, True)
        img = np.array(img, dtype=float)
        print('<> ' + file + ' compressed successfully!')
        out_path = './SyedAbbasOutputs/' + file[:-4] + 'Compressed.tif'
        plt.imsave(out_path, img, cmap='gray', format='TIFF')
        print('<> ' + file[:-4] + 'Compressed.tif' + ' saved successfully!\n')
