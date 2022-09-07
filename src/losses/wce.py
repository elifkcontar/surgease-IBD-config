from typing import List

import numpy as np
import torch
import torch.nn as nn


class multitask_WCE(torch.nn.Module):
    def __init__(
        self, weights: List[np.ndarray], reduction="mean", device="cpu"
    ):
        super(multitask_WCE, self).__init__()
        self.num_multitask = len(weights)

        self.criterions = [
            nn.CrossEntropyLoss(
                weight=torch.tensor(w).float().to(device), reduction=reduction
            )
            for w in weights
        ]

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ):
        target = target.t()
        l0 = self.criterions[0](input[0], target[0])
        l1 = self.criterions[1](input[1], target[1])
        l2 = self.criterions[2](input[2], target[2])

        loss = l0 + l1 + l2

        return loss, (l0, l1, l2)

class multitask_MSE_ordinal(torch.nn.Module):
    def __init__(
        self, weights: List[np.ndarray], reduction="mean", device="cpu"
    ):
        super(multitask_MSE_ordinal, self).__init__()
        self.num_multitask = len(weights)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ):
        target = target.t()
        # Create out modified target with [batch_size, num_labels] shape
        modified_target = torch.zeros_like(input[0])

        # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
        for i, targ in enumerate(target[0]):
            modified_target[i, 0:targ+1] = 1

        l0 = nn.MSELoss(reduction='none')(input, modified_target).sum(axis=1)
        #-------------------------------------
        modified_target = torch.zeros_like(input[1])

        for i, targ in enumerate(target[1]):
            modified_target[i, 0:targ+1] = 1

        l1 = nn.MSELoss(reduction='none')(input, modified_target).sum(axis=1)
        #-----------------------------------------
        modified_target = torch.zeros_like(input[1])

        for i, targ in enumerate(target[1]):
            modified_target[i, 0:targ+1] = 1

        l2 = nn.MSELoss(reduction='none')(input, modified_target).sum(axis=1)
        #--------------------------------------------

        loss = l0 + l1 + l2

        return loss, (l0, l1, l2)