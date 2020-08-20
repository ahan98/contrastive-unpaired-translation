import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from training.train import train

def test():
    in_N, in_C, in_H, in_W = 1, 3, 256, 256  # note we assume 256x256 images
    encoder_fn = Encoder()
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    out, samples = encoder_fn(sample_in)

if __name__ == "__main__":
    test()
