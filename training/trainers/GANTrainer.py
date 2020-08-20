import torch

class GANTrainer:

    @staticmethod
    def train_discriminator(generator, discriminator, solver, real_data):
        """
        Trains discriminator on real and fake data, and computes its loss.

        Inputs:
        - [nn.Module] generator - generator network
        - [nn.Module] discriminator - discriminator network
        - [torch.optim] solver - SGD optimizer
        - [torch.Tensor] real_data - tensor of sample from real dataset

        Returns discriminator loss.
        """

        solver.zero_grad()  # reset gradients

        # generate fake data
        # NOTE: we detach generator since it is fixed during discriminator
        # training
        noise = torch.randn(real_data.shape)
        fake_data = generator(noise).detach()

        # train on real and fake data
        prediction_real = discriminator(real_data)
        prediction_fake = discriminator(fake_data)
        loss = __discriminator_loss(prediction_real, prediction_fake)

        loss.backward()
        solver.step()

        return loss

    @staticmethod
    def train_generator(generator, discriminator, solver, real_data_shape):
        """
        Trains generator on fake data, and computes its loss.

        Inputs:
        - [nn.Module] generator - generator network
        - [nn.Module] discriminator - discriminator network
        - [torch.optim] solver - SGD optimizer
        - [torch.Size] real_data_shape - shape of sample from real dataset

        Returns generator loss, and the image generated from noise (to be passed
        into the PatchNCE network).
        """

        solver.zero_grad()  # reset gradients

        # generate fake data
        noise = torch.randn(real_data_shape)
        fake_data = generator(noise)  # here, we DO want loss to backprop

        # train on fake data only
        prediction_fake = discriminator(fake_data)
        loss = __generator_loss(prediction_fake)

        loss.backward()
        solver.step()

        return loss, fake_data

    ''' Private Static Methods '''

    @staticmethod
    def __discriminator_loss(predictions_real, predictions_fake):
        """
        Inputs:
        - [torch.Tensor] predictions_real: output of discriminator given sample
          from real dataset
        - [torch.Tensor] predictions_fake: output of discriminator given
          generated sample from noise

        Returns least squares loss. See section 3.2 of Mao et al.
        """
        loss = torch.mean((predictions_real - 1)**2) + torch.mean(predictions_fake**2)
        return 0.5 * loss

    @staticmethod
    def __generator_loss(predictions_fake):
        """
        Inputs:
        - [torch.Tensor] predictions_fake: output of discriminator given
          generated sample from noise

        Returns least squares loss.
        """
        loss = torch.mean((predictions_fake - 1)**2)
        return 0.5 * loss
