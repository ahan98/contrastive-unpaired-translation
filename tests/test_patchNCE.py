import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.PatchNCE.PatchNCE import PatchNCE
from models.GAN.Encoder import Encoder

def test():
    in_N, in_C, in_H, in_W = 1, 3, 256, 256  # note we assume 256x256 images
    encoder = Encoder()
    patch_nce_fn = PatchNCE(encoder)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    features_final = patch_nce_fn(sample_in)

    n_layers_sampled = 5
    if len(features_final) != n_layers_sampled:
        raise RuntimeError(
            ("Incorrect number of layers sampled.\n"
             + "Expected {}, got {}")
            .format(n_layers_sampled, len(features_final))
        )

    sample_size = 256
    for layer_key in features_final:
        feature_tensor = features_final[layer_key]
        out_N = feature_tensor.shape[0]
        out_feature_size = feature_tensor.shape[2]

        if (out_N != in_N) or (out_feature_size != sample_size):
            raise RuntimeError(
                ("Incorrect sample shape.\n"
                 + "Expected {}, got {}")
                .format(expected_shape, feature_tensor.shape)
            )

    print("Encoder test passed")
    return True

if __name__ == "__main__":
    test()
