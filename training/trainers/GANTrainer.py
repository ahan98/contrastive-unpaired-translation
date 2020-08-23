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
        solver.zero_grad()  # reset gradients
        criterion = torch.nn.MSELoss()

        # Train on real Data
        prediction_real = discriminator(real_data)
        target_real = torch.ones(prediction_real.shape, device=device)
        # Calculate error and backpropagate
        loss_real = GANTrainer.criterion(prediction_real, target_real)
        loss_real.backward()

        # Generate fake data
        noise = torch.randn(real_data.shape, device=device)
        fake_data = generator(noise).detach()

        # Train on fake data
        prediction_fake = discriminator(fake_data)
        target_fake = torch.zeros(prediction_fake.shape, device=device)
        # Calculate error and backpropagate
        loss_fake = GANTrainer.criterion(prediction_fake, target_fake)
        loss_fake.backward()

        solver.step()  # update parameters

        return loss_real + loss_fake

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
        fake_data = generator(noise)  # here, we DO want loss to backprop

        # Train on fake data (only)
        prediction_fake = discriminator(fake_data)
        target_fake = torch.zeros(prediction_fake.shape, device=device)
        # Calculate error and backpropagate
        loss_fake = GANTrainer.criterion(prediction_fake, target_fake)
        loss_fake.backward()

        solver.step()

        return loss_fake, fake_data
