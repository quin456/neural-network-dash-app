
import torch
from torch import nn


class Test(nn.Module):

    def __init__(self):
        super().__init__()
        custom_layer = nn.Linear(3,3)
        weights = torch.tensor([
            [1, 2, 3],
            [2, 3, 4],
            [4, 5, 6]
        ])
        with torch.no_grad():
            custom_layer.weight.fill_(weights)
        self.layers = nn.Sequential(
            custom_layer
        )

tester = Test()