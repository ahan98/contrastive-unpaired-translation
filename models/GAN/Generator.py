import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder
from blocks.typing_ import PaddingMode

class Generator(nn.Module):

    def __init__(self, n_res_blocks=9, batch_momentum=0.1, padding_mode=PaddingMode.REFLECT):
        super().__init__()
        self.encoder = Encoder(n_res_blocks, batch_momentum, padding_mode)
        self.decoder = Decoder(n_res_blocks, batch_momentum, padding_mode)

    def forward(self, x):
        out, _ = self.encoder(x)
        out = self.decoder(out)
        return out
