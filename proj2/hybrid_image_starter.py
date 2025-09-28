#starter code
import matplotlib.pyplot as plt
from align_image_code import align_images

import numpy as np
from scipy.signal import convolve2d
import skimage.transform as sktr
import cv2
import math

# First load images

# high sf
im1 = plt.imread('./derek_nutmeg/DerekPicture.jpg')/255.0

# low sf
im2 = plt.imread('./derek_nutmeg/nutmeg.jpg')/255.0

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

#do everything in grayscale for now
#im1_grayscale = np.mean(im1, axis=2) if len(im1.shape) > 2 else im1
#im2_grayscale = np.mean(im2, axis=2) if len(im2.shape) > 2 else im2

im1_grayscale = np.mean(im1_aligned, axis=2)
im2_grayscale = np.mean(im2_aligned, axis=2)


print ("checking alignment befre making a hybrid: ")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(im1_aligned)
plt.title('Aligned im1')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(im2_aligned)
plt.title('Aligned im2')
plt.axis('off')
plt.show()

plt.imsave('./im1_aligned.jpg', im1_aligned)
plt.imsave('./im2_aligned.jpg', im2_aligned)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

'''
SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns esentially:
1. Low-pass filter the first image to remove all high frequencies, keep only smooth areas
- LP = Original * Gaussian
- visible from FAR AWAY

2. High pass filter the 2nd image to remove low freqs and keep all the high edges
- HP = Original - LP 
- visble from UP CLOSE

3. Put 1 and 2 together into a hybrid image
'''
def create_gaussian(sigma):

    ksize = 2 * math.ceil(3 * sigma) + 1 #common formula for gaussian kernel size (.ceil rounds to nearest int)
    ksize = ksize + 1 if ksize%2 == 0 else ksize #kernel needs to be odd sized

    gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d.T)
    return gaussian_2d

def hybrid_image(im1, im2, sigma1, sigma2):

    #im1(low pass) and im2(high pass) need their own separate kernelss
    low_im1_kernel = create_gaussian(sigma1)
    high_im2_kernel = create_gaussian(sigma2)

    # low_im1 = np.zeros_like(im1)
    # high_im2 = np.zeros_like(im2)

    #low pass filter on im1 = get rid of high freqs = visble from far
    low_im1 = convolve2d(im1, low_im1_kernel, mode='same', boundary='fill', fillvalue=0)

    #high pass filter on im2 = get rid of low freqs = visible from close
    low_im2_pass = convolve2d(im2, high_im2_kernel, mode='same', boundary='fill', fillvalue=0)
    high_im2 = im2 - low_im2_pass

    # for channel in range(3):
    #     low_im1[:, :, channel] = convolve2d(im1[:, :, channel], low_im1_kernel, mode='same', boundary='fill', fillvalue=0)

    #     low_im2_pass = convolve2d(im2[:, :, channel], high_im2_kernel, mode='same', boundary='fill', fillvalue=0)
    #     high_im2[:, :, channel] = im2[:, :, channel] - low_im2_pass

    hybrid = low_im1 + high_im2
    hybrid = np.clip(hybrid, 0, 1)

    return hybrid, low_im1, high_im2

'''
For your favorite result, you should also illustrate the process through frequency analysis. 
Show the log magnitude of the Fourier transform of the two input images, the filtered images, and the hybrid image. 
In Python, you can compute and display the 2D Fourier transform with: plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image)))))
'''
def freq_analysis(images, titles):
    plt.figure(figsize=(15, 8))

    for i, (image, title) in enumerate(zip(images, titles)):
        gray = np.mean(image, axis=2) if image.ndim == 3 else image
        fourier_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))))

        plt.subplot(2, 3, i+1)
        plt.imshow(fourier_spectrum, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


sigma1 = 12 #derek
sigma2 = 9 #nutmeg

print(f"Hybrid image with sigma1 = {sigma1} and sigma2 = {sigma2}")
hybrid, low_im1, high_im2 = hybrid_image(im1_grayscale, im2_grayscale, sigma1, sigma2)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(im1_grayscale, cmap='gray')
plt.title('Im1 original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(im2_grayscale, cmap='gray')
plt.title('Im2 original')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(hybrid, cmap='gray')
plt.title(f"Hybrid image with sigma1 = {sigma1} and sigma2 = {sigma2}")
plt.axis('off')

plt.tight_layout()
plt.show()

print("Frequency analysis: ")
images = [im1_grayscale, im2_grayscale, low_im1, high_im2, hybrid]
titles = ['Im1 original', 'Im2 original', 'Low Pass Im1', 'High Pass im2', 'Hybrid']

freq_analysis(images, titles)