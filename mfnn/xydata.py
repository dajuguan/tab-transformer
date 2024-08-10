#!/usr/bin/env python3

from __future__ import annotations
import typing
import torch

TensorFunc = typing.Callable[[torch.Tensor], torch.Tensor]

class XYZDataSet(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        if len(x) != len(y):
            raise ValueError('size of x and y not match')
        self.x = x
        self.y = y
        self.z = z

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.z[idx]

class XYDataSet(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        if len(x) != len(y):
            raise ValueError('size of x and y not match')
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]