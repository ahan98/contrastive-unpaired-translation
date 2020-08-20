import torch
from .utils import *

def train_D(D, solver, real_data):
    solver.zero_grad()  # reset gradients

    # generate fake data
    noise = make_noise(real_data.shape)
    fake_data = G(noise).detach()  # here, G is fixed

    # train on real data
    prediction_real = D(real_data)
    target_real = real_data_target(real_data.shape)
    loss_real = loss_D(prediction_real, target_real)
    loss_real.backward()

    # train on fake data
    prediction_fake = D(fake_data)
    target_fake = fake_data_target(real_data.shape)
    loss_fake = loss_D(prediction_fake, target_fake)
    loss_fake.backward()

    solver.step()

    return loss_real + loss_fake, prediction_real, prediction_fake


def train_G(G, D, solver, real_data_shape):
    solver.zero_grad()  # reset gradients

    # generate fake data
    noise = make_noise(real_data_shape)
    fake_data = G(noise)  # here, we DO want loss to backprop

    # train on fake data, using real targets, i.e., measure how well generator
    # can fool discriminator
    prediction = D(fake_data)
    target = real_data_target(real_data_shape)
    loss = loss_G(prediction, target)
    loss.backward()

    solver.step()

    return loss, fake_data


def loss_D(real_data, fake_data):
    loss = torch.mean((real_data - 1)**2) + torch.mean(fake_data**2)
    return -0.5 * loss


def loss_G(fake_data):
    loss = torch.mean((fake_data - 1)**2)
    return -0.5 * loss
