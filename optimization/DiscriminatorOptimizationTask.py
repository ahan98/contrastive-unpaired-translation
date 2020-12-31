from typing import Any
import bbml.training as training
import bbml.DataPersisting
import torch.optim


class DiscriminatorOptimizationTask(training.OptimizationTask):

    def __init__(self,
                 generator: training.TrainableModel,
                 discriminator: training.TrainableModel,
                 discriminator_optimizer: training.ModelOptimizer):
        super().__init__()

        self.__generator = generator
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer

    def learn_mapping(self, x: Any, y: Any, context: training.TrainingSessionContext):

        self.discriminator.set_requires_grad(True)
        self.discriminator_optimizer.zero_grad()

        # discriminator wants to learn to tell these apart
        real_y = y
        fake_y = self.__generator(x).detach()

        prediction_real = self.discriminator(real_y)
        target_real = torch.ones(prediction_real.shape, device=context.device())
        loss_real = torch.nn.MSELoss(prediction_real, target_real).mean()

        prediction_fake = self.discriminator(fake_y)
        target_fake = torch.zeros(prediction_fake.shape, device=context.device())
        loss_fake = torch.nn.MSELoss(prediction_fake, target_fake).mean()

        total_loss = 0.5 * (loss_real + loss_fake)
        total_loss.backward()
        # TODO: this might be asynchronous??
        self.discriminator_optimizer.step()

        self.discriminator.set_requires_grad(False)

    def save_with_data_persister(self, directory: str, data_perister: bbml.DataPersisting):
        data_perister.save_state(self.discriminator, directory, self.discriminator.identifier())
        data_perister.save_state(self.discriminator_optimizer, directory, self.discriminator_optimizer.identifier())

    def load_with_data_persister(self, directory: str, data_persister: bbml.DataPersisting):
        discriminator_dict = data_persister.load_state("{}/{}".format(directory, self.discriminator.identifier()))
        self.discriminator.load_state_from_dictionary(discriminator_dict)

        discriminator_optimizer_dict = data_persister.load_state("{}/{}".format(directory, self.discriminator_optimizer.identifier()))
        self.discriminator_optimizer.load_state_from_dictionary(discriminator_optimizer_dict)