import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = '#171717'
plt.rcParams['text.color'] = '#DDDDDD'


def single_plot():
    # Open the HDF5 file
    with h5py.File('data/content/data/volume_1_slice_16.h5', 'r') as file:
        # List all groups (keys) in the file
        print("Keys in the file:", list(file.keys()))

        # Load the image and mask datasets
        image = file['image'][:]
        mask = file['mask'][:]

    # Check the shape of the image and mask
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    # Plot the image and mask
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    for i in range(image.shape[-1]):  # Loop through the 4 channels
        axes[i].imshow(image[:, :, i], cmap='gray')
        axes[i].set_title(f'Image Channel {i + 1}')
        axes[i].axis('off')

    plt.figure()
    for i in range(mask.shape[-1]):  # Loop through the 3 mask channels
        plt.subplot(1, 3, i + 1)
        plt.imshow(mask[:, :, i], cmap='gray')
        plt.title(f'Mask Channel {i + 1}')
        plt.axis('off')

    plt.show()


def display_image_channels(image, title="Image Channels"):
    channel_names = [
        'T1-weighted (T1)',
        'T1-weighted post contrast (T1c)',
        'T2-weighted (T2)',
        'Fluid Attenuated Inversion Recovery (FLAIR)'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for idx, ax in enumerate(axes.flatten()):
        channel_image = image[idx, :, :]
        ax.imshow(channel_image, cmap='magma')
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.suptitle(title, fontsize=20, y=1.03)
    plt.show()


def display_mask(mask, title="Mask"):
    channel_names = ['Necrotic (NEC)', 'Edema (ED)', 'Tumour (ET)']
    fig, axes = plt.subplots(1, 3, figsize=(9.75, 5))
    for idx, ax in enumerate(axes):
        rgb_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        rgb_mask[..., idx] = mask[idx, :, :] * 255
        ax.imshow(rgb_mask)
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.suptitle(title, fontsize=20, y=0.93)
    plt.tight_layout()
    plt.show()


def overlay_masks_on_image(image, mask, title='Brain MRI with Tumour Masks Overlay'):
    t1_image = image[0, :, :]  # Use the first channel of the image
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.stack([t1_image_normalized] * 3, axis=-1)
    color_mask = np.stack([mask[0, :, :], mask[1, :, :], mask[2, :, :]], axis=-1)
    rgb_image = np.where(color_mask, color_mask, rgb_image)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title(title, fontsize=18, y=1.02)
    plt.axis('off')
    plt.show()


def display_slice(image, slice_index, axis=0):
    plt.figure(figsize=(8, 8))
    if axis == 0:  # Axial slice (depth axis)
        plt.imshow(image[slice_index, :, :], cmap='magma')
    elif axis == 1:  # Coronal slice (height, axis)
        plt.imshow(image[:, slice_index, :], cmap='magma')
    elif axis == 2:  # Sagittal slice (width axis)
        plt.imshow(image[:, :, slice_index], cmap='magma')
    plt.title(f'Slice {slice_index} (Axis {axis})')
    plt.axis('off')
    plt.show()


def overlay_slice(image, mask, slice_index, axis=0):
    plt.figure(figsize=(6, 6))
    if axis == 0:  # Axial slice
        plt.imshow(image[slice_index, :, :], cmap='gray')
        plt.imshow(mask[slice_index, :, :], cmap='Reds', alpha=0.5)
    elif axis == 1:  # Coronal slice
        plt.imshow(image[:, slice_index, :], cmap='gray')
        plt.imshow(mask[:, slice_index, :], cmap='Reds', alpha=0.5)
    elif axis == 2:  # Sagittal slice
        plt.imshow(image[:, :, slice_index], cmap='gray')
        plt.imshow(mask[:, :, slice_index], cmap='Reds', alpha=0.5)
    plt.title(f'Overlay Slice {slice_index} (Axis {axis})')
    plt.axis('off')
    plt.show()


def visualize_predictions(model, data_loader, device, num_images=5):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Thresholding

            if i == num_images:
                break

            # Plot the original image, ground truth mask, and predicted mask
            plt.figure(figsize=(15, 5))

            # Squeeze the first dimension to remove the channel dimension for visualization
            plt.subplot(1, 3, 1)
            plt.imshow(images[0, 0, :, :].squeeze().cpu().numpy(), cmap='gray')
            plt.title('Original Image')

            plt.subplot(1, 3, 2)
            plt.imshow(masks[0, 0, :, :].squeeze().cpu().numpy(), cmap='gray')
            plt.title('Ground Truth Mask')

            plt.subplot(1, 3, 3)
            plt.imshow(preds[0, 0, :, :].squeeze().cpu().numpy(), cmap='gray')
            plt.title('Predicted Mask')

            plt.show()
