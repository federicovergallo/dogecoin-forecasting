import torch.nn.functional as F
from torch.nn import MSELoss


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    loss = MSELoss()
    return loss(output.float(), target.float())
