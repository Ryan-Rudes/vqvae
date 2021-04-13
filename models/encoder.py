
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Encoder(nn.Module):
  def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
    super(Encoder, self).__init__()

    self.conv_stack = nn.Sequential(
        nn.Conv2d(in_dim, 32, 8, 4),
        nn.ReLU(),
        nn.Conv2d(32, 128, 4, 2),
        nn.ReLU(),
        nn.Conv2d(128, 512, 3, 2),
        ResidualStack(
            512, 512, res_h_dim, n_res_layers)

    )

  def forward(self, x, log=False):
    return self.conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
