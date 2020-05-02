import torch
from skimage import data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

bird = mpimg.imread("datasets/images/bird.jpeg")
plt.title('Original Image')
plt.imshow(bird)

bird.shape
bird[200: 250, 200:250]

bird_reshape = bird.reshape(bird.shape[0], -1)
bird_reshape.shape

plt.figure(figure = (6, 6))
plt.title('Reshaped image')
plt.imshow(bird_reshape)

bird_resized = skimage.transform.resize(bird, (500, 500))
bird_resized.shape

plt.figure(figure = (6, 6))
plt.title('Resized image')
plt.imshow(bird_resized)

aspect_ratio_original = bird.shape[1] / float(bird.shape[0])
aspect_ratio_resized = bird_resized.shape[1] / float(bird_resized.shape[0])
print('Original aspect ratio', aspect_ratio_original)
print('Resized aspect ratio', aspect_ratio_resized)

bird_rescaled = skimage.transform.rescale(bird_resized, (1.0, aspect_ratio_original))
bird_rescaled.shape

plt.figure(figure = (6, 6))
plt.title('Rescaled image')
plt.imshow(bird_rescaled)

aspect_ratio_rescaled = bird_rescaled[1] / float(bird_rescaled[0])
print('Rescaled aspect ratio', aspect_ratio_rescaled)

bird_BGR = bird[:, :, (2, 1, 0)]
plt.figure(figure = (6, 6))
plt.title('BGR(Blue Green Red) image')
plt.imshow(bird_BGR)

bird_gray = skimage.color.rgb2gray(bird)
plt.figure(figure = (6, 6))
plt.title('Gray image')
plt.imshow(bird_gray, cmap = 'gray')
bird_gray.shape

giraffes = skimage.img_as_float(skimage.io.imread('datasets/images/giraffes.jpg')).astype(np.float32)
plt.figure(figure = (6, 6))
plt.title('Original image')
plt.imshow(giraffes)
giraffes.shape

def crop(image, cropx, cropy):
    y, x, c = image.shape
    startx = x//2 - (cropx // 8)
    straty = y//3 - (cropy // 4)
    stopx = startx + cropx
    stopy = starty + 2*cropy
    return image[starty:stopy, startx:stopx]

giraffes_cropped = crop(giraffes, 256, 256)
plt.figure(figure = (6, 6))
plt.title('Cropeed image')
plt.imshow(giraffes_cropped)

from skiage.util import random_noise

sigma = 0.155
noisy_giraffes = random_noise(giraffes, var = sigma**2)
plt.figure(figure = (6, 6))
plt.title('Image with added noise')
plt.imshow(noisy_giraffes)

from sklearn.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma

sigma_est = estimate_sigma(noisy_giraffes, multichannel = True, average_sigmas = True)
sigma_est
plt.imshow(denoise_tv_chambolle(noisy_giraffes))
plt.imshow(denoise_bilateral(noisy_giraffes, sigma_color = 0.05, sigma_spatial = 15, multichannel = True))
plt.imshow(denoise_wavelet(noisy_giraffes, multichannel = True))

monkeys = skimage.img_as_float(skimage.io.imread('datasets/images/monkeys.jpg')).astype(np.float32)
plt.figure(figure = (6, 6))
plt.title('Original image')
plt.imshow(monkeys)

monkeys_flip = np.fliplr(monkeys)
plt.figure(figure = (6, 6))
plt.title('Horizontal flip')
plt.imshow(monkeys_flip)

mirror = skimage.img_as_float(skimage.io.imread('datasets/images/book-mirrored.jpg')).astype(np.float32)
plt.figure(figure = (6, 6))
plt.title('Original image')
plt.imshow(mirror)

mirror_flip = np.fliplr(mirror)
plt.figure(figure = (6, 6))
plt.title('Horizontal flip')
plt.imshow(mirror_flip)
