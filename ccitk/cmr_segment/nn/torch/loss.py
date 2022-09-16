import math
from typing import Union, Iterable

import torch
import torch.nn.functional as F


class TorchLoss(torch.nn.Module):
    """Wrapper for torch loss"""

    def __init__(self):
        super().__init__()
        self._count = 0
        self._cum_loss = 0

    def description(self):
        """Description use for display"""
        return "{}: {:.4f}".format(self.document(), self.avg())

    def log(self):
        """Value(s) to store in logging"""
        return self.avg()

    def document(self):
        """Document what loss this is and hyper params if any"""
        return self.__class__.__name__

    def cumulate(
        self,
        predicted: Union[torch.Tensor, Iterable[torch.Tensor]],
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
    ):
        loss = self(predicted, outputs)
        self._cum_loss += loss.item()
        self._count += 1
        return loss

    def reset(self):
        self._cum_loss = 0
        self._count = 0

    def avg(self):
        if self._count > 0:
            return self._cum_loss / self._count
        else:
            return 0

    def new(self):
        """Copy and reset obj"""
        new_loss = self.__class__()
        new_loss.reset()
        return new_loss


class FocalLoss(TorchLoss):
    def __init__(self, alpha: float = 1, gamma: float = 2, logits: bool = False, reduce: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, predicted, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(predicted, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(predicted, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

    def description(self):
        return "Focal loss: {:.4f}".format(self.avg())

    def document(self):
        return "Focal Loss - alpha: {}, gamma: {}, logits: {}, reduce: {}".format(
            self.alpha, self.gamma, self.logits, self.reduce
        )

    def new(self):
        new_loss = self.__class__(gamma=self.gamma, alpha=self.alpha, reduce=self.reduce, logits=self.logits)
        new_loss.reset()
        return new_loss


class DiceCoeff(TorchLoss):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        eps = 0.0001
        num = input.size(0)
        m1 = input.view(num, -1).float()  # Flatten
        m2 = target.view(num, -1).float()  # Flatten
        self.inter = (m1 * m2).sum().float()
        self.union = m1.sum() + m2.sum()
        t = (2 * self.inter.float() + eps) / (self.union.float() + eps)
        return t


class DiceLoss(TorchLoss):
    """
    N-D dice for segmentation
    """

    def forward(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class MSELoss(TorchLoss, torch.nn.MSELoss):
    def description(self):
        return "MSE loss: {:.4f}".format(self.avg())


class L1Loss(TorchLoss, torch.nn.L1Loss):
    pass


class BCELoss(TorchLoss):
    def __init__(self, logit: bool = True):
        super().__init__()
        self.logit = logit

    def forward(self, predicted, targets):
        if self.logit:
            BCE_loss = F.binary_cross_entropy_with_logits(predicted, targets, reduction="mean")
        else:
            BCE_loss = F.binary_cross_entropy(predicted, targets, reduction="mean")
        return BCE_loss


class DiceCoeffWithLogits(DiceCoeff):
    def forward(self, logits, target):
        pred = torch.sigmoid(logits)
        pred = (pred > 0.5).float()
        return super().forward(pred, target)
