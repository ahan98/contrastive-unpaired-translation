import torch


class GANTrainer:

    criterion = torch.nn.MSELoss()

    @staticmethod
    def train_discriminator(generator, discriminator, solver, real_data,
                            device="cpu"):
        """
        Trains discriminator on real and fake data, and returns its loss.

        Inputs:
        - [nn.Module] generator - generator network
        - [nn.Module] discriminator - discriminator network
        - [torch.optim] solver - gradient optimizer
        - [torch.Tensor] real_data - tensor of sample from real dataset
        """

        # Reset gradients and allow backprop to discriminator
        solver.zero_grad()

        # Train on real data
        GANTrainer.set_requires_grad(discriminator, True)
        prediction_real = discriminator(real_data)
        target_real = torch.ones(prediction_real.shape, device=device)
        loss_real = GANTrainer.criterion(prediction_real, target_real)

        # Generate fake data
        noise = torch.randn(real_data.shape, device=device)
        fake_data = generator(noise).detach()  # NOTE: don't backprop generator

        # Train on fake data
        prediction_fake = discriminator(fake_data)
        target_fake = torch.zeros(prediction_fake.shape, device=device)
        loss_fake = GANTrainer.criterion(prediction_fake, target_fake)

        # Compute gradients and update parameters
        loss_average = 0.5 * (loss_real + loss_fake)
        loss_average.backward()
        solver.step()

        return loss_average.item()

    @staticmethod
    def train_generator(generator, discriminator, solver, real_data_shape,
                        device="cpu"):
        """
        Trains generator on fake data, and computes its loss.

        Inputs:
        - [nn.Module] generator - generator network
        - [nn.Module] discriminator - discriminator network
        - [torch.optim] solver - gradient optimizer
        - [torch.Size] real_data_shape - shape of sample from real dataset

        Returns generator loss, and the image generated from noise (to be passed
        into the PatchNCE network).
        """

        solver.zero_grad()  # reset gradients

        # Generate fake data
        noise = torch.randn(real_data_shape, device=device)
        fake_data = generator(noise)  # NOTE: allow backprop for generator

        # Train on fake data (only)
        GANTrainer.set_requires_grad(discriminator, False)
        prediction_fake = discriminator(fake_data)
        target_fake = torch.zeros(prediction_fake.shape, device=device)
        # Calculate error and backpropagate
        loss_fake = GANTrainer.criterion(prediction_fake, target_fake)
        loss_fake.backward()

        solver.step()

        return loss_fake.item(), fake_data

    @staticmethod
    def set_requires_grad(network, requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad
