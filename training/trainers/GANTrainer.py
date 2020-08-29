import torch


class GANTrainer:

    criterion = torch.nn.MSELoss()

    @staticmethod
    def train_discriminator(discriminator, real_data, fake_data, device="cpu"):
        """
        Trains discriminator on real and fake data, and returns its loss.

        Inputs:
        - [nn.Module] generator - generator network
        - [nn.Module] discriminator - discriminator network
        - [torch.optim] solver - gradient optimizer
        - [torch.Tensor] real_data - tensor of sample from real dataset
        """

        # Train on real data
        prediction_real = discriminator(real_data)
        target_real = torch.ones(prediction_real.shape, device=device)
        loss_real = GANTrainer.criterion(prediction_real, target_real).mean()

        # Train on fake data
        prediction_fake = discriminator(fake_data)
        target_fake = torch.zeros(prediction_fake.shape, device=device)
        loss_fake = GANTrainer.criterion(prediction_fake, target_fake).mean()

        # Average real and fake loss
        weighted_average_loss = 0.5 * (loss_real + loss_fake)

        return weighted_average_loss

    @staticmethod
    def train_generator(discriminator, fake_data, device="cpu"):
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

        # Train on fake data (only)
        prediction_fake = discriminator(fake_data)
        target_fake = torch.zeros(prediction_fake.shape, device=device)

        # Calculate fake loss
        loss_fake = GANTrainer.criterion(prediction_fake, target_fake).mean()

        return loss_fake
