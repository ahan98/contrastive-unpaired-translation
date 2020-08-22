import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.GAN.Decoder import Decoder

def test():
    in_N, in_C, in_H, in_W = 1, 256, 64, 64
    decoder_fn = Decoder()
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    out = decoder_fn(sample_in)

    expected_shape = in_N, 3, in_H * 4, in_W * 4

    if out.shape != expected_shape:
        raise RuntimeError(
            ("Incorrect output shape.\n"
             + "Expected output shape: {}\n"
             + "Actual output shape: {}")
            .format(expected_shape, out.shape)
        )

    print("Decoder test passed")
    return True

if __name__ == "__main__":
    test()
