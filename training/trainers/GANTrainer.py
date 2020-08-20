import torch
from .utils import make_noise

class GANTrainer:

    @staticmethod
    def train_discriminator(generator, discriminator, solver, real_data, device):
        solver.zero_grad()  # reset gradients

        # generate fake data
        noise = make_noise(real_data.shape)
        fake_data = generator(noise).detach()  # here, G is fixed

        # training on real data
        prediction_real = discriminator(real_data)
        target_real = GANTrainer.__real_data_target(real_data.shape, device)
        loss_real = GANTrainer.__discriminator_loss(prediction_real, target_real)
        loss_real.backward()

        # training on fake data
        prediction_fake = discriminator(fake_data)
        target_fake = GANTrainer.__fake_data_target(real_data.shape, device)
        loss_fake = GANTrainer.__discriminator_loss(prediction_fake, target_fake)
        loss_fake.backward()

        solver.step()

        return loss_real + loss_fake, prediction_real, prediction_fake

    @staticmethod
    def train_generator(generator, discriminator, solver, real_data_shape, device):
        solver.zero_grad()  # reset gradients

        # generate fake data
        noise = make_noise(real_data_shape)
        fake_data = generator(noise)  # here, we DO want loss to backprop

        # training on fake data, using real targets, i.e., measure how well generator
        # can fool discriminator
        prediction = discriminator(fake_data)
        target = GANTrainer.__real_data_target(real_data_shape, device)
        loss = GANTrainer.__generator_loss(prediction)
        loss.backward()

        solver.step()

        return loss, fake_data

    ''' Private Static Methods '''

    @staticmethod
    def __discriminator_loss(real_data, fake_data):
        loss = torch.mean((real_data - 1)**2) + torch.mean(fake_data**2)
        return -0.5 * loss

    @staticmethod
    def __generator_loss(fake_data):
        loss = torch.mean((fake_data - 1)**2)
        return -0.5 * loss

    @staticmethod
    def __real_data_target(shape, device):
        '''
        Tensor containing ones, with shape = size
        '''
        target = torch.ones(shape, device=device)
        return target

    @staticmethod
    def __fake_data_target(shape, device):
        '''
        Tensor containing zeros, with shape = size
        '''
        target = torch.zeros(shape, device=device)
        return target