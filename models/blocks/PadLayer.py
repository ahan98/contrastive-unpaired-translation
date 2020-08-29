import torch.nn as nn
from .types import PaddingMode, padding_mode_to_str


class PadLayer(nn.Module):
    def __init__(self, padding=1, padding_mode=PaddingMode.REFLECT):
        super().__init__()

        padding_mode_str = padding_mode_to_str(padding_mode)

        if padding_mode == PaddingMode.REFLECT:
            self.model = nn.ReflectionPad2d(padding)
        elif padding_mode == PaddingMode.ZEROS:
            self.model = nn.ZeroPad2d(padding)
        elif padding_mode == PaddingMode.REPLICATE:
            self.model = nn.ReplicationPad2d(padding)
        else:
            raise NotImplementedError("Padding mode {} not implemented"
                                      .format(padding_mode_str))

    def forward(self, x):
        out = self.model(x)
        return out
