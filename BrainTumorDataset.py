import torch
import h5py
from torch.utils.data import Dataset
import torchvision.transforms as T


class BrainTumorDataset(Dataset):
    def __init__(self, h5_files, target_shape=(240, 240), transform=None, augment=False):
        self.h5_files = h5_files
        self.transform = transform
        self.target_shape = target_shape
        self.augment = augment

        # Initialize the resize transformation
        self.resize = T.Resize(target_shape)

        # Define augmentations if augment is True
        if self.augment:
            self.augmentation_transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(30),
                T.RandomResizedCrop(target_shape, scale=(0.8, 1.0))
            ])
        else:
            self.augmentation_transforms = None

    def z_score_normalization(self, image, epsilon=1e-8):
        return (image - image.mean()) / (image.std() + epsilon)

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, index):
        h5_file = self.h5_files[index]

        with h5py.File(h5_file, 'r') as f:
            image = f['image'][()]
            mask = f['mask'][()]

            # Apply normalization
            image = self.z_score_normalization(image)
            mask = self.z_score_normalization(mask)

            # Convert to tensors and permute dimensions
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Shape: (C, H, W)
            mask = torch.tensor(mask, dtype=torch.long).permute(2, 0, 1)  # Shape: (C, H, W)

            # Apply augmentations if augment is True
            if self.augment:
                # Combine the image and mask into a single tensor for joint transformation
                combined = torch.cat([image, mask], dim=0)  # Shape: (C+C, H, W)
                combined = self.augmentation_transforms(combined)
                image = combined[:image.size(0)]  # Separate the augmented image
                mask = combined[image.size(0):]  # Separate the augmented mask

            # Resize the images and masks
            image = self.resize(image)
            mask = self.resize(mask)

            # Add an additional dimension if necessary (if working with single-channel data)
            image = image.unsqueeze(1)  # Shape: (C, 1, H, W)
            mask = mask.unsqueeze(1)  # Shape: (C, 1, H, W)

        # Apply additional transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        return image, mask
