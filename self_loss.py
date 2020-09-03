import torch
import torch.nn as nn
from config import *

class Encoder_Loss(nn.Module):
    def __init__(self, lambda_param=10, dim=2):
        super(Encoder_Loss, self).__init__()
        self.dim = dim

    def forward(self, mu, sigma):
        return -0.5 * torch.sum(1+sigma-mu**2-torch.exp(sigma))

