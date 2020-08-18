import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.PatchNCE.PatchNCE import PatchNCE

def test():
    in_N, in_C, in_H, in_W = 1, 3, 256, 256  # note we assume 256x256 images
    patch_nce_fn = PatchNCE()
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    features_final = patch_nce_fn(sample_in)

    n_layers_sampled = 5
    if len(features_final) != n_layers_sampled:
        raise RuntimeError(
            "Incorrect number of layers sampled. Expected {}, got {}"
            .format(n_layers_sampled, len(features_final))
        )

    expected_shape = torch.Size([256])
    for layer_key in features_final:
        feature_tensor = features_final[layer_key]
        if feature_tensor.shape != expected_shape:
            raise RuntimeError(
                "Incorrect sample shape. Expected {}, got {}"
                .format(expected_shape, feature_tensor.shape)
            )

    print("Encoder test passed")
    return True

if __name__ == "__main__":
    test()