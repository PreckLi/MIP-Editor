from torch import nn
import torch
from typing_extensions import override

class PartialLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, trainable_cols: list[int]):
        super().__init__()
        self.original_linear = original_linear
        self.num_rows = original_linear.in_features
        self.num_columns = original_linear.out_features

        self.trainable_cols = sorted(trainable_cols)
        self.frozen_cols = [
            i for i in range(self.num_columns) if i not in trainable_cols
        ]

        self.trainable_linear = nn.Linear(
            self.num_rows, len(self.trainable_cols), bias=False
        )
        self.trainable_linear.weight = nn.Parameter(original_linear.weight[self.trainable_cols, :].clone())
        self.trainable_linear.requires_grad_(True)
        self.trainable_linear.to(original_linear.weight.device)

        self.frozen_linear = nn.Linear(
            self.num_rows, len(self.frozen_cols), bias=False
        )
        self.frozen_linear.weight = nn.Parameter(original_linear.weight[self.frozen_cols, :].clone())
        self.frozen_linear.requires_grad_(False)
        self.frozen_linear.to(original_linear.weight.device)

        self.is_weight_changed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.is_weight_changed = True
            trainable_output = self.trainable_linear(x)
            frozen_output = self.frozen_linear(x)
            output = torch.zeros(
                *x.shape[:-1], self.num_columns, device=x.device, dtype=x.dtype
            )
            output[..., self.trainable_cols] = trainable_output
            output[..., self.frozen_cols] = frozen_output
            return output
        else:
            if self.is_weight_changed:
                self.merge_to_linear()
                self.is_weight_changed = False
            return self.original_linear(x)
    
    def merge_to_linear(self):
        """Merge the trainable and frozen parts back into the original linear layer."""
        with torch.no_grad():
            self.original_linear.weight[self.trainable_cols, :] = self.trainable_linear.weight
            self.original_linear.weight[self.frozen_cols, :] = self.frozen_linear.weight
        return self.original_linear


