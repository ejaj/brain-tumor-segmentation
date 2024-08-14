import random
import scipy.ndimage as ndimage
import cv2
import numpy as np


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    # Check if the min and max values are the same
    if min_val == max_val:
        # If they are the same, return a zeroed array or a normalized array with all zeros
        return np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / (max_val - min_val)
        return normalized_image


def resize_image(image, target_shape=(128, 128), interpolation=cv2.INTER_LINEAR):
    resized_image = np.zeros((image.shape[0], *target_shape))
    for i in range(image.shape[0]):  # Loop through each slice
        resized_image[i] = cv2.resize(image[i], target_shape, interpolation=interpolation)
    return resized_image


def augment_image(image, mask):
    if random.random() > 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)  # Apply same flip to mask

    if random.random() > 0.5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=2)  # Apply same flip to mask

    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        image = ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=1)
        mask = ndimage.rotate(mask, angle, axes=(1, 2), reshape=False, order=0)  # Nearest neighbor for masks

    return image, mask
