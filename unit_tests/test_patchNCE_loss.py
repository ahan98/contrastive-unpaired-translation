import torch
from training.trainers.PatchNCETrainer import PatchNCETrainer


def test():
    N, C, S = 1, 64, 256
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    feat_x = torch.zeros((N, C, S), device=device)
    feat_gx = torch.zeros((N, C, S), device=device)
    print("Using device", device)
    loss = PatchNCETrainer._patchNCE_loss(
        feat_x, feat_gx, verbose=True, reduction="none", device=device)

    expected_shape = torch.Size([N * S])
    if loss.shape != expected_shape:
        raise RuntimeError("Incorrect loss shape. Expected {}, got {}"
                           .format(expected_shape, loss.shape))

    print("PatchNCE loss test passed")
    return True


if __name__ == "__main__":
    test()
