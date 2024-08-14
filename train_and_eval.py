import torch
import wandb
from tqdm import tqdm

# Initialize wandb
wandb.init(project='brain-tumor-segmentation', name='model_training_v1')


def log_metrics(epoch, train_loss, val_loss, learning_rate):
    wandb.log({
        "Epoch": epoch,
        "Training Loss": train_loss,
        "Validation Loss": val_loss,
        "Learning Rate": learning_rate
    })


def dice_coefficient(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)  # Convert logits to probabilities
    preds = (preds > threshold).float()  # Apply threshold
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection) / (preds.sum() + targets.sum())
    return dice.item()


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for images, masks in tqdm(train_loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        masks = masks.float()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_dice += dice_coefficient(outputs, masks)

    epoch_loss = running_loss / len(train_loader)
    epoch_dice = running_dice / len(train_loader)

    # Log to wandb and tqdm
    wandb.log({"Training Loss": epoch_loss, "Training Dice": epoch_dice})
    tqdm.write(f'Training Loss: {epoch_loss:.4f}, Training Dice: {epoch_dice:.4f}')

    return epoch_loss, epoch_dice


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_dice += dice_coefficient(outputs, masks)

    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_dice = val_dice / len(val_loader)

    # Log to wandb and tqdm
    wandb.log({"Validation Loss": epoch_val_loss, "Validation Dice": epoch_val_dice})
    tqdm.write(f'Validation Loss: {epoch_val_loss:.4f}, Validation Dice: {epoch_val_dice:.4f}')

    return epoch_val_loss, epoch_val_dice


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device,
                save_path='best_model.pth', early_stopping_patience=5):
    best_val_loss = float('inf')
    patience_counter = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        # Train step
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validation step
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

        # Print the losses and Dice coefficient
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {train_loss:.4f}, '
              f'Training Dice: {train_dice:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Dice: {val_dice:.4f}'
              )

        # Update the learning rate if scheduler is provided
        if scheduler:
            scheduler.step(val_loss)

        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the counter if validation loss improves
            filename = f"best_model_epoch_{epoch + 1:02d}_val_loss_{val_loss:.4f}_dice_{val_dice:.4f}.pth"
            torch.save(model.state_dict(), filename)
            print(f'Best model saved as {filename}')
        else:
            patience_counter += 1

        # If the patience counter exceeds the patience threshold, stop training
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered. Training stopped.")
            break

    print("Training complete!!")

# Testing (Evaluation) Function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_dice = 0.0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing", leave=False):
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            test_dice += dice_coefficient(outputs, masks)

    avg_test_loss = test_loss / len(test_loader)
    avg_test_dice = test_dice / len(test_loader)

    # Log to wandb and tqdm
    wandb.log({"Test Loss": avg_test_loss, "Test Dice": avg_test_dice})
    tqdm.write(f'Test Loss: {avg_test_loss:.4f}, Test Dice: {avg_test_dice:.4f}')

    return avg_test_loss, avg_test_dice
