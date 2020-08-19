import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from loss.PatchNCELoss import PatchNCELoss

def test():
    N, C, S = 1, 64, 256
    feat_x = torch.zeros((N, C, S))
    feat_gx = torch.zeros((N, C, S))
    loss_fn = PatchNCELoss()
    loss = loss_fn(feat_x, feat_gx, verbose=True)

    expected_shape = torch.Size([N * S])
    if loss.shape != expected_shape:
        raise RuntimeError("Incorrect loss shape. Expected {}, got {}"
                           .format(expected_shape, loss.shape))

    print("PatchNCE loss test passed")
    return True

if __name__ == "__main__":
    test()
