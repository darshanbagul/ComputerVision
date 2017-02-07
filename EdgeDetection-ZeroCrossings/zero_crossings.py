import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import math

DoG_kernel = [
    [0,   0, -1, -1, -1, 0, 0],
    [0,  -2, -3, -3, -3,-2, 0],
    [-1, -3,  5,  5,  5,-3,-1],
    [-1, -3,  5, 16,  5,-3,-1],
    [-1, -3,  5,  5,  5,-3,-1],
    [0,  -2, -3, -3, -3,-2, 0],
    [0,   0, -1, -1, -1, 0, 0]
]

LoG_kernel = np.array([
    [0, 0,  1, 0, 0],
    [0, 1,  2, 1, 0],
    [1, 2,-16, 2, 1],
    [0, 1,  2, 1, 0],
    [0, 0,  1, 0, 0]
])

def plot_input(img, title):
    plt.imshow(img, cmap = 'gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()
    
def handle_img_padding(img1, img2):
    M1, N1 = img1.shape[:2]
    M2, N2 = img2.shape[:2]
    padding_x = np.abs(M2 - M1)/2
    padding_y = np.abs(N2 - N1)/2
    img2 = img2[padding_x:M1+padding_x, padding_y: N1+padding_y]
    return img2

def zero_cross_detection(image):
    z_c_image = np.zeros(image.shape)

    for i in range(0,image.shape[0]-1):
        for j in range(0,image.shape[1]-1):
            if image[i][j]>0:
                if image[i+1][j] < 0 or image[i+1][j+1] < 0 or image[i][j+1] < 0:
                    z_c_image[i,j] = 1
            elif image[i][j] < 0:
                if image[i+1][j] > 0 or image[i+1][j+1] > 0 or image[i][j+1] > 0:
                    z_c_image[i,j] = 1
    return z_c_image

def calculate_sobel_edges(original_img):
    sobelx = cv2.Sobel(original_img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(original_img,cv2.CV_64F,0,1,ksize=3)
    sobel_first_derivative = cv2.magnitude(sobelx,sobely)
    return sobel_first_derivative


def threshold_first_derivative_edges(sobel_test):
    sobel_test[sobel_test > 200] = 255
    sobel_test[sobel_test < 200] = 0
    return sobel_test


def logical_and_two_images(img1, img2):
    return cv2.bitwise_and(img1, img2)


def DoG_solution(original_img):
    dog_img = convolve2d(original_img, DoG_kernel)
    plot_input(dog_img, 'DoG Image')

    zero_crossing_dog = zero_cross_detection(dog_img)
    zero_crossing_dog = handle_img_padding(original_img, zero_crossing_dog)
    plot_input(zero_crossing_dog,'DoG - Zero Crossing')

    sobel_first_derivative = calculate_sobel_edges(original_img)
    sobel_first_derivative = threshold_first_derivative_edges(sobel_first_derivative)
    plot_input(sobel_first_derivative,'Boosted 1st order Derivative - Sobel edges')

    DoG_solution_img = logical_and_two_images(zero_crossing_dog, sobel_first_derivative)
    plot_input(DoG_solution_img, 'Strong Edges detected by DoG Zero Crossing')


def LoG_solution(original_img):
    log_img = convolve2d(original_img, LoG_kernel)
    plot_input(log_img, 'LoG Image')

    zero_crossing_log = zero_cross_detection(log_img)
    zero_crossing_log = handle_img_padding(original_img, zero_crossing_log)
    plot_input(zero_crossing_log,'Zero Crossing-LoG')

    sobel_first_derivative = calculate_sobel_edges(original_img)
    sobel_first_derivative = threshold_first_derivative_edges(sobel_first_derivative)
    plot_input(sobel_first_derivative,'Boosted 1st order Derivative - Sobel edges')

    LoG_solution_img = logical_and_two_images(zero_crossing_log, sobel_first_derivative)
    plot_input(LoG_solution_img, 'Strong edges detected by LoG Zero Crossing')

def main():
    original_img = cv2.imread('./UBCampus.jpg', 0)
    plot_input(original_img, 'Original Image')

    DoG_solution(original_img)

    LoG_solution(original_img)

if __name__ == "__main__":
    main()