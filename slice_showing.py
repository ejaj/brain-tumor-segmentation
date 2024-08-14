import numpy as np
import os
import h5py
from plots import display_slice, overlay_slice

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
    # Display the middle slice of the volume along the axial plane
    if 'image' in data:
        slice_index = image.shape[0] // 2  # Middle slice
        display_slice(image, slice_index, axis=0)

    # Display the middle slice of the volume along the coronal plane
    if 'image' in data:
        slice_index = image.shape[1] // 2  # Middle slice
        display_slice(image, slice_index, axis=1)

    # Display the middle slice of the volume along the sagittal plane
    if 'image' in data:
        slice_index = image.shape[2] // 2  # Middle slice
        display_slice(image, slice_index, axis=2)

    # Display a range of slices along the axial plane
    if 'image' in data:
        for i in range(0, image.shape[0], 10):
            display_slice(image, i, axis=0)

    # Overlay of a single slice
    if 'image' in data and 'mask' in data:
        image = data['image'].transpose((2, 0, 1))
        mask = data['mask'].transpose((2, 0, 1))
        slice_index = image.shape[0] // 2  # Middle slice
        overlay_slice(image, mask, slice_index, axis=0)
else:
    print("No .h5 files found in the directory.")
