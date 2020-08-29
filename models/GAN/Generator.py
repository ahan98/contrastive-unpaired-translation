import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder
from ..blocks.types import PaddingMode


class Generator(nn.Module):

    def __init__(self, encoder=None, decoder=None, n_res_blocks=9,
                 batch_momentum=0.1, padding_mode=PaddingMode.REFLECT):

        super().__init__()

        if encoder is None:
            encoder = Encoder(n_res_blocks, padding_mode=padding_mode)

        if decoder is None:
            decoder = Decoder(padding_mode)

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        out, _ = self.encoder(x)
        out = self.decoder(out)
        return out
