import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        smooth = 1.0  # To prevent division by zero
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return bce_loss + dice_loss
