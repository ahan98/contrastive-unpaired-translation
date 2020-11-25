from typing import Optional
import torch.optim
import bbml.nn
import bbml.models.training as training
from .Encoder import Encoder
from .Decoder import Decoder


class Generator(training.TrainableModel):

    def __init__(self,
                 optimizer: torch.optim.optimizer,
                 encoder: Optional[torch.nn.Module] = None,
                 decoder: Optional[torch.nn.Module] = None,
                 n_res_blocks: int = 9,
                 batch_momentum: float = 0.1,
                 padding_mode: bbml.PaddingMode = bbml.PaddingMode.REFLECT):

        if encoder is None:
            encoder = Encoder(n_res_blocks, padding_mode=padding_mode, batch_momentum=batch_momentum)

        if decoder is None:
            decoder = Decoder(padding_mode, batch_momentum=batch_momentum)

        self.encoder = encoder
        self.decoder = decoder

        super().__init__(optimizer)

    def forward(self, x, encode_only: bool = False):
        out, samples = self.encoder(x)
        if encode_only:
            return samples
        out = self.decoder(out)
        return out
