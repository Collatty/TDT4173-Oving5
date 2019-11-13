import cv2 as cv
import matplotlib.pyplot as plt
import random
import skimage as sk
from skimage import util, transform

image = cv.imread('./detection-images/detection-1.jpg')
plt.imshow(image, cmap="gray")
plt.show()


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)



def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array, mode='s&p')


def horizontal_flip(image_array):
    # flip image
    return image_array[:, ::-1]


image = horizontal_flip(image)
plt.imshow(image, cmap="gray")
plt.show()
