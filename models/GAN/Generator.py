import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder
from blocks.types import PaddingMode

class Generator(nn.Module):

    def __init__(self, encoder=None, decoder=None, n_res_blocks=9, batch_momentum=0.1, padding_mode=PaddingMode.REFLECT):
        super().__init__()

        if encoder is None:
            encoder = Encoder(n_res_blocks, batch_momentum, padding_mode)

        if decoder is None:
            decoder = Decoder(n_res_blocks, batch_momentum, padding_mode)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        out, _ = self.encoder(x)
        out = self.decoder(out)
        return out
