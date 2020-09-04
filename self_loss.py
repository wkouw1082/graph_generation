import torch
import torch.nn as nn
from config import *

class Encoder_Loss(nn.Module):
    def __init__(self):
        super(Encoder_Loss, self).__init__()

    def forward(self, mu, sigma):
        return -0.5 * torch.sum(1+sigma-mu**2-torch.exp(sigma))

