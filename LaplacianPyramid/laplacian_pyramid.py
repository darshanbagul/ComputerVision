import numpy as np
import cv2
import matplotlib.pyplot as plt

gaussian_kernel = (1.0/16)*np.array([[1, 2, 1],[2,4,2],[1,2,1]])


def preprocess_odd_images(img):
    M, N = img.shape[:2]
    if M % 2 == 1 and N % 2 == 1:
        return img[1:][1:]
    elif M%2 == 1:
        return img[1:][:]
    elif N%2 == 1:
        return img[:][1:]
    else:
        return img

def convolve(f, g):
    vmax = f.shape[0]
    wmax = f.shape[1]
    smax = g.shape[0]
    tmax = g.shape[1]
    smid = smax // 2
    tmid = tmax // 2
    xmax = vmax + 2*smid
    ymax = wmax + 2*tmid
    # Allocate result image.
    h = np.zeros([xmax, ymax], dtype=f.dtype)
    # Do convolution
    for x in range(xmax):
        for y in range(ymax):
            # Calculate pixel value for h at (x,y). Sum one component
            # for each pixel (s, t) of the filter g.
            s_from = max(smid - x, -smid)
            s_to = min((xmax - x) - smid, smid + 1)
            t_from = max(tmid - y, -tmid)
            t_to = min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h

def interpolate(image):
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    image_up[::2, ::2] = image
    return convolve(image_up, 4*gaussian_kernel)


def gaussian_pyramid(image):
    image_blur = convolve(image, gaussian_kernel)
    return image_blur[::2, ::2]

def plot_input(img, title):
    plt.imshow(img, cmap = 'gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def sub_plot(img1, img2, img3, title):
    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(img1, cmap = 'gray')
    plt.subplot(132)
    plt.imshow(img2, cmap = 'gray')
    plt.subplot(133)
    plt.imshow(img3, cmap = 'gray')
    plt.show()


def handle_img_padding(img1, img2):
    M1, N1 = img1.shape[:2]
    M2, N2 = img2.shape[:2]
    padding_x = np.abs(M2 - M1)/2
    padding_y = np.abs(N2 - N1)/2
    img2 = img2[padding_x:M1+padding_x, padding_y: N1+padding_y]
    return img2

def min_sqr_err(mat1, mat2):
    min_sq = 0
    M, N = mat1.shape[:2]
    for i in range(M):
        for j in range(N):
            min_sq += np.square(mat1[i][j] - mat2[i][j])
    return min_sq

def reconstruct_original_img(G, L):
    reconstructed_images = []
    for i in range(len(G)-1 , 0, -1):
        expanded_img = interpolate(G[i])
        if expanded_img.shape[:2]!= L[i-1].shape[:2]:
            resized_img = handle_img_padding(L[i-1], expanded_img)
        reconstructed_images.append(resized_img + L[i-1])
        # sub_plot(expanded_img, L[i-1], resized_img + L[i-1], "Levels" + str(i))
    return reconstructed_images

def create_gaussian_laplacian_pyramids(image, level):
    G = [image]
    L = []
    while level > 0:
        level -= 1
        image_blur = gaussian_pyramid(image)
        G.append(image_blur)
        expanded_img = interpolate(image_blur)
        if image.shape[:2] != expanded_img.shape[:2]:
            expanded_img = handle_img_padding(image, expanded_img)
        laplacian = image - expanded_img
        L.append(laplacian)
        image = image_blur
    return G, L


def main():
    original_img = cv2.imread('./lena_sodberg.jpg', 0)
    plot_input(original_img, 'Level0')
    original_img = preprocess_odd_images(original_img)
    M,N = original_img.shape[:2]

    pyramid_levels = 6     # 6 Gaussian Levels, inclusive of original image, lead to 5 Laplacian levels.     

    gaussian_pyramid, laplacian_pyramid = create_gaussian_laplacian_pyramids(original_img, pyramid_levels)

    for i in range(len(laplacian_pyramid)-1, 0, -1):
        plot_input(laplacian_pyramid[i], 'Laplacian Pyramid - Level '+ str(i))

    reconstructed_images = reconstruct_original_img(gaussian_pyramid, laplacian_pyramid)
    final_image = reconstructed_images[-1]
    plot_input(original_img, 'Original Image')
    plot_input(final_image, 'Reconstructed Image')

    print "Minimum Squared Error (MSE) : ", min_sqr_err(original_img, final_image)

if __name__ == '__main__':
    main()