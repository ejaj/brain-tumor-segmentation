import cv2
import numpy as np
import os
import h5py
from plots import display_image_channels, display_mask, overlay_masks_on_image, display_slice, overlay_slice
from preprocess import normalize_image, resize_image, augment_image
directory = "data/content/data/"

# Create a list of all .h5 files in the directory
h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
print(f"Found {len(h5_files)} .h5 files:\nExample file names:{h5_files[:3]}")

if h5_files:
    file_path = os.path.join(directory, h5_files[25071])
    data = {}
    with h5py.File(file_path, 'r') as f:
        print("\nKeys for each file:", list(f.keys()))
        for key in f.keys():
            data[key] = f[key][()]
            print(f"\nData type of {key}:", type(f[key][()]))
            print(f"Shape of {key}:", f[key].shape)
            print(f"Array dtype: {f[key].dtype}")
            print(f"Array max val: {np.max(f[key])}")
            print(f"Array min val: {np.min(f[key])}")

    print(data['image'].shape)
    image = data['image'].transpose((2, 0, 1))
    mask = data['mask'].transpose(2, 0, 1)
    # display_image_channels(image)
    # display_mask(mask)
    # overlay_masks_on_image(image, mask)
    # normalize_image = normalize_image(data['image'])
    # display_image_channels(normalize_image)

    # Normalize image (not necessary for mask)
    image = normalize_image(image)
    # Resize both image and mask
    target_shape = (128, 128)  # Example target size
    image = resize_image(image, target_shape=target_shape)
    mask = resize_image(mask, target_shape=target_shape,
                        interpolation=cv2.INTER_NEAREST)
    # Augment both image and mask
    image, mask = augment_image(image, mask)

    display_image_channels(image)
    display_mask(mask)
else:
    print("No .h5 files found in the directory.")
