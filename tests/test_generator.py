import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.GAN.Generator import Generator

def test():
    in_N, in_C, in_H, in_W = 1, 3, 256, 256
    generator_fn = Generator()
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    out = generator_fn(sample_in)

    expected_shape = sample_in.shape

    if out.shape != expected_shape:
        raise RuntimeError(
            ("Incorrect output shape.\n"
             + "Expected output shape: {}\n"
             + "Actual output shape: {}")
            .format(expected_shape, out.shape)
        )

    print("Generator test passed")
    return True

if __name__ == "__main__":
    test()
