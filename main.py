import re
from collections import defaultdict

import os
import re
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from BrainTumorDataset import BrainTumorDataset
from Loss import CombinedLoss
from UNet3D import UNet3D
from plots import visualize_predictions
from train_and_eval import evaluate_model, train_model

directory = "data/content/data/"

# Create a list of all .h5 files in the directory
h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
h5_files = [os.path.abspath(f) for f in h5_files]  # Ensure absolute paths

print(f"Found {len(h5_files)} .h5 files:\nExample file names:{h5_files[:3]}")

# Group files by volume number
volume_groups = defaultdict(list)
for h5_file in h5_files:
    # Extract volume number using regex
    volume_number = re.findall(r'volume_(\d+)_', os.path.basename(h5_file))[0]
    volume_groups[volume_number].append(h5_file)

# Convert the grouped files to a list of lists
volumes = list(volume_groups.values())
print(f"Total volumes: {len(volumes)}")

# Define the proportions for training, validation, and testing
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Calculate the number of volumes for each set
total_volumes = len(volumes)  # 369 in this case
train_size = int(train_split * total_volumes)
val_size = int(val_split * total_volumes)
test_size = total_volumes - train_size - val_size  # Ensures all data is used

# Split the volumes into training, validation, and test sets
train_volumes, val_volumes, test_volumes = random_split(volumes, [train_size, val_size, test_size])

# Flatten the lists of volumes back into a list of slices
train_files = [file for volume in train_volumes for file in volume]
val_files = [file for volume in val_volumes for file in volume]
test_files = [file for volume in test_volumes for file in volume]

print(f"Training set size: {len(train_files)} slices")
print(f"Validation set size: {len(val_files)} slices")
print(f"Test set size: {len(test_files)} slices")

# Create the datasets
train_dataset = BrainTumorDataset(train_files)
val_dataset = BrainTumorDataset(val_files)
test_dataset = BrainTumorDataset(test_files)

# Create DataLoaders for each set
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model with 4 input channels and 1 output channel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for images, masks in train_loader:
    images, masks = images.to(device), masks.to(device)

    # Verify input shape before passing to model
    print(f"Input shape to model: {images.shape}")  # Should be (batch_size, 4, depth, height, width)
    break
model = UNet3D(in_channels=4, out_channels=3).to(device)

criterion = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Train the model
train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
    device=device,
    early_stopping_patience=5  # Stop if no improvement for 5 epochs
)
saved_model_path = 'best_model_epoch_01_val_loss_1.6847_dice_0.0074.pth'
model.load_state_dict(torch.load(saved_model_path, weights_only=False))
model.eval()

# Evaluate the model on the test set
test_loss, test_dice = evaluate_model(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}')

# Visualize predictions on the test set
visualize_predictions(model, test_loader, device)
