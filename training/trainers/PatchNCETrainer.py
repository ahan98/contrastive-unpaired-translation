import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy


class PatchNCETrainer:
    @staticmethod
    def train_patchnce(patchNCE, solver, real_data, fake_data, device="cpu"):
        """
        Trains PatchNCE network on real and fake data, and computes its loss.

        Inputs:
        - [nn.Module] patchNCE: PatchNCE network
        - [torch.optim] solver: SGD optimizer
        - [torch.Tensor] real_data: tensor of sample from real dataset
        - [torch.Tensor] fake_data: tensor of sample generated from noise
        - [String] device: name of device to load data (e.g., "cuda:0")

        Returns:
        - [float] PatchNCE loss
        """

        solver.zero_grad()

        # Extract features from layers of the encoder sub-network, and compute
        # loss for each layer's features.
        real_features_dict = patchNCE(real_data)
        fake_features_dict = patchNCE(fake_data)
        loss = 0
        for layer_key in real_features_dict:
            real_features = real_features_dict[layer_key]
            fake_features = fake_features_dict[layer_key]
            loss += PatchNCETrainer._patchNCE_loss(real_features, fake_features,
                                                   device)

        loss.backward()
        solver.step()

        average_loss_per_layer = loss / len(real_features_dict)

        return average_loss_per_layer

    @staticmethod
    def _patchNCE_loss(feat_x, feat_gx, device="cpu", tau=0.07,
                       reduction="mean", verbose=False):
        """
        Computes the patchwise contrastive loss between sampled feature maps
        from H(G_enc(x)) and sampled feature maps from H(G_enc(G(x))).
        Intuitively, we can think of each spatial location (or matrix cell) of a
        feature map as corresponding to a patch in x and G(x), where patches in
        the deeper feature map layers correspond to larger patches in x and
        G(x). Then the patchwise contrastive loss computes the (softmax) cross
        entropy of cosine similarities between the generated and positive input
        patches, and the generated and negative input patches.

        See equations 2, 3, and 5 of the paper.

        Note that the original input image may be sampled from either the domain
        x (e.g., horse) or domain y (e.g., zebra). If sampled from y, the loss
        acts as a regularizer, stabilizing changes made to the generator.
        Intuitively, this loss will be low for a generator that already works
        well, and high for a bad generator. So once a generator works well,
        further change is discouraged.

        Inputs:
        - [torch.Tensor] feat_x: sampled feature maps from H(G_enc(x))
        - [torch.Tensor] feat_gx: sampled feature maps from H(G_enc(G(x)))
        - [float] tau: temperature parameter to scale logits
        - [string] reduction: mode to reduce cross entropy loss
        - [bool] verbose: if True, prints expected shape of predictions and loss
        - [String] device: name of device to load data (e.g., "cuda:0")

        Returns:
        - [float] PatchNCE loss
        """

        # N - batch size (default 1)
        # C - num. channels
        # S - sample size (default 256)
        N, C, S = feat_x.shape

        # Detach feat_gx so that backpropagation only affects the encoder half
        # of G. See section C.5., item 1, which explains that reusing the same
        # encoder-affine network both stabilizes training, and reduces the
        # number of parameters in the network.
        feat_gx = feat_gx.detach()

        # Compute the cosine similarity score (i.e., dot product) between the
        # "fake" patches (i.e., derived from G) and the corresponding real
        # patches. Each dot product involves two length-C vectors, one
        # corresponding to a patch/cell from feat_x, and the other from feat_gx.
        # Intuitively, the cosine similarity will be larger the more similar two
        # patches are.
        pwc_pos = (feat_x * feat_gx).sum(dim=1)[:, :, None]   # (N,S,1)

        # Compute cosine similarity between the fake patches and all negative
        # patches. If the similarity scores between two patches has shape
        # (N,S,1), then the similarity scores between the fake patch and a
        # vector of negative patches has shape (N,S,S), where the columns of
        # each SxS matrix denote the similarity scores between one negative
        # patch and the output patch.
        pwc_negs = torch.bmm(feat_x.transpose(1, 2), feat_gx)  # (N,S,S)

        # Broadcast SxS mask across all N matrices, where the diagonal is a
        # negative number that "zeros" out the diagonals of pwc_negs, since each
        # diagonal represents a feature's similarity with itself, which is
        # meaningless. Note that these diagonals are are not literally zeroed
        # out, but rather become a near-zero, positive value after
        # exponentiation from softmax cross entropy.
        mask = torch.eye(S, dtype=torch.bool, device=device)[
            None, :, :]  # (1,S,S)
        pwc_negs.masked_fill_(mask, float("-inf"))

        # torch.cat appends each length-S column vector in pwc_pos to the
        # corresponding SxS matrix in pwc_negs, resulting in shape (N,S,S+1).
        logits = torch.cat((pwc_pos, pwc_negs), dim=2) / tau

        # Compute the cross entropy loss, where the positive patch similarity is
        # stored in index 0 of each of the (N*S) vectors.
        predictions = logits.flatten(0, 1)  # (N,S,S+1) -> (N*S, S+1)
        targets = torch.zeros(N * S, dtype=torch.long, device=device)

        loss = cross_entropy(predictions, targets, reduction=reduction)

        if verbose:
            print("Shape of predictions: {}\nExpected: {}"
                  .format(predictions.shape, torch.Size([N*S, S+1])))
            print("Shape of cross entropy loss: {}\nExpected: {}"
                  .format(loss.shape, torch.Size([N*S])))

        return loss
