import cv2
import numpy as np
import matplotlib.pyplot as plt
import cmath

def preprocess_odd_images(img):
    ''' If we have an odd sized image, we convert it into an even sized image. Necessary for centering the Fourier Transform'''
    M, N = img.shape[:2]
    if M % 2 == 1 and N % 2 == 1:
        return img[1:][1:]
    elif M%2 == 1:
        return img[1:][:]
    elif N%2 == 1:
        return img[:][1:]
    else:
        return img

def img_preprocess_centred_fourier(img):
    M, N = img.shape[:2]
    processed_img = np.zeros((M,N))
    for x in range(M/2):
        for y in range(N/2):
            processed_img[x][y] = img[M/2 + x][N/2 + y]
    for x in range(M/2+1, M):
        for y in range(N/2+1, N):
            processed_img[x][y] = img[x - M/2 ][y - N/2]
    for x in range(M/2+1, M):
        for y in range(N/2):
            processed_img[x][y] = img[x - M/2][y + N/2]
    for x in range(M/2):
        for y in range(N/2+1, N/2):
            processed_img[x][y] = img[x + M/2][y - N/2]
    return processed_img


def two_dim_fourier_transform(img):
    M, N = img.shape[:2]
    dft_rep = [[0.0 for k in range(M)] for l in range(N)]
    for k in range(M):
        for l in range(N):
            temp_sum = 0.0
            for m in range(M):
                for n in range(N):
                    e = cmath.exp(- 1j * 2 * cmath.pi * (float(k * m) / M + float(l * n) / N))
                    temp_sum += img[m][n] * e
            dft_rep[l][k] = temp_sum
    return dft_rep


def two_dim_inv_fourier_transform(fourier):
    M = len(fourier)
    N = len(fourier[0])
    idft_rep = [[0.0 for k in range(M)] for l in range(N)]
    for k in range(M):
        for l in range(N):
            temp_sum = 0.0
            for m in range(M):
                for n in range(N):
                    e = cmath.exp(1j * 2 * cmath.pi * (float(k * m) / M + float(l * n) / N))
                    temp_sum += fourier[m][n] * e
            idft_rep[l][k] = temp_sum/(M*N)
    return idft_rep


def plot_input(img, title):
    plt.imshow(img, cmap = 'gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


def min_sqr_err(mat1, mat2):
    min_sq = 0
    M, N = mat1.shape[:2]
    for i in range(M):
        for j in range(N):
            min_sq += np.square(mat1[i][j] - mat2[i][j])
    return min_sq


def fourier_transforms():
    img = cv2.imread('./fb_test.jpg', 0)
    img = preprocess_odd_images(img)
    M, N = img.shape[:2]
    print "Size of image: ", M,N
    processed_img = img_preprocess_centred_fourier(img)
    # Plot function calls have been commented. Uncomment to view the output.
    plot_input(img, 'Original Image')
    plot_input(processed_img, 'Processed Image')
    
    dft_rep = two_dim_fourier_transform(img)
    centred_dft_rep = two_dim_fourier_transform(processed_img)
    idft_rep = two_dim_inv_fourier_transform(dft_rep)

    plot_input(10*np.log(1+np.abs(centred_dft_rep)), 'Centred Discrete Fourier Transform')
    plot_input(np.abs(idft_rep), 'Inverse Fourier Transform')

    min_sq_error = min_sqr_err(img, np.abs(idft_rep))
    print "Mean squared Error between the reconstructed and original image: ", min_sq_error

if __name__ == '__main__':
    fourier_transforms()