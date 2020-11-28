from typing import Any
import bbml.models.training as training
import torch.optim


class DiscriminatorOptimizationTask(training.OptimizationTask):

    def __init__(self,
                 generator: training.TrainableModel,
                 discriminator_ensemble: training.ModelOptimizerEnsemble):
        super().__init__()

        self.__generator = generator
        self.__discriminator_ensemble = discriminator_ensemble

    def learn_mapping(self, x: Any, y: Any, context: training.TrainingSessionContext):
        discriminator = self.__discriminator_ensemble.model()
        optimizer = self.__discriminator_ensemble.optimizer()

        discriminator.set_requires_grad(True)
        optimizer.zero_grad()

        # discriminator wants to learn to tell these apart
        real_y = y
        fake_y = self.__generator(x).detach()

        prediction_real = discriminator(real_y)
        target_real = torch.ones(prediction_real.shape, device=context.device())
        loss_real = torch.nn.MSELoss(prediction_real, target_real).mean()

        prediction_fake = discriminator(fake_y)
        target_fake = torch.zeros(prediction_fake.shape, device=context.device())
        loss_fake = torch.nn.MSELoss(prediction_fake, target_fake).mean()

        total_loss = 0.5 * (loss_real + loss_fake)
        total_loss.backward()
        # TODO: this might be asynchronous??
        optimizer.step()

        discriminator.set_requires_grad(False)
