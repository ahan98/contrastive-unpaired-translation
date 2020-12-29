from typing import Any
import torch.optim
from torch.nn.functional import cross_entropy
import bbml.models.training as training


class GeneratorOptimizationTask(training.OptimizationTask):

    def __init__(self, discriminator: training.TrainableModel,
                 generator: training.TrainableModel,
                 patchNCE: training.TrainableModel,
                 generator_optimizer: training.ModelOptimizer,
                 patchNCE_optimizer: training.ModelOptimizer):
        super().__init__()

        self.__discriminator = discriminator
        self.generator = generator
        self.patchNCE = patchNCE

        self.generator_optimizer = generator_optimizer
        self.patchNCE_optimizer = patchNCE_optimizer

    def learn_mapping(self, x: Any, y: Any, context: training.TrainingSessionContext):
        # Generator Loss
        self.generator.zero_grad()
        fake_y = self.generator(x)
        gen_loss = self.__generator_loss(x, context)

        # PatchNCE Loss
        self.patchNCE.zero_grad()
        patchNCE_loss_x = self.__patchNCE_loss(x, fake_y, context)

        fake_y = self.generator(x)
        patchNCE_loss_y = self.__patchNCE_loss(y, fake_y, context)

        patchNCE_loss_avg = 0.5 * (patchNCE_loss_x + patchNCE_loss_y)

        # Learn
        total_loss = patchNCE_loss_avg + gen_loss
        total_loss.backward()

        self.generator_optimizer.step()
        self.patchNCE_optimizer.step()

    ''' PRIVATE '''

    def __generator_loss(self, fake_y: Any, context: training.TrainingSessionContext) -> Any:
        prediction_fake = self.__discriminator(fake_y)
        target_fake = torch.zeros(prediction_fake.shape, device=context.device())
        return torch.nn.MSELoss(prediction_fake, target_fake).mean()

    def __patchNCE_loss(self, real_y: Any, fake_y: Any, context: training.TrainingSessionContext) -> Any:

        real_samples = self.generator(real_y, encode_only=True)
        fake_samples = self.generator(fake_y, encode_only=True)

        feat_x = self.patchNCE(real_samples)
        feat_gx = self.patchNCE(fake_samples)

        total_nce_loss = 0
        for idx, sample in enumerate(real_samples):
            real_sample = feat_x[idx]
            fake_sample = feat_gx[idx]
            total_nce_loss += self.__patchNCE_loss_per_sample(fake_sample, real_sample, context).mean()

        average_nce_loss = total_nce_loss / len(feat_x)
        return average_nce_loss

    def __patchNCE_loss_per_sample(self, feat_gx, feat_x, context: training.TrainingSessionContext, tau=0.07) -> Any:
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

        batch_size = feat_x.shape[0]
        dim = feat_x.shape[1]
        feat_gx = feat_gx.detach()

        # pos logit
        l_pos = torch.bmm(feat_x.view(batch_size, 1, -1),
                          feat_gx.view(batch_size, -1, 1))
        l_pos = l_pos.view(batch_size, 1)

        # neg logit -- current batch
        # reshape features to batch size
        feat_x = feat_x.view(1, -1, dim)
        feat_gx = feat_gx.view(1, -1, dim)
        npatches = feat_x.size(1)
        l_neg_curbatch = torch.bmm(feat_x, feat_gx.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=context.device(),
                             dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / tau

        loss = cross_entropy(out, torch.zeros(out.size(0),
                                              dtype=torch.long,
                                              device=context.device()))
        return loss
