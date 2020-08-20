from torch import randn, mean

def train_D(D, solver, real_data):
    """
    Trains discriminator on real and fake data, and computes its loss.

    Inputs:
    - [nn.Module] D - discriminator network
    - [torch.optim] solver - SGD optimizer
    - [torch.Tensor] real_data - tensor of sample from real dataset

    Returns discriminator loss.
    """
    solver.zero_grad()  # reset gradients

    # generate fake data
    # NOTE: we detach G because for the training of D, G is consider fixed and
    # should not be updated by backprop
    noise = randn(real_data.shape)
    fake_data = G(noise).detach()

    # train on real and fake data
    prediction_real = D(real_data)
    prediction_fake = D(fake_data)
    loss = loss_D(prediction_real, prediction_fake)

    loss.backward()
    solver.step()

    return loss


def train_G(G, D, solver, real_data_shape):
    """
    Trains generator on fake data, and computes its loss.

    Inputs:
    - [nn.Module] G - generator network
    - [nn.Module] D - discriminator network
    - [torch.optim] solver - SGD optimizer
    - [torch.Size] real_data_shape - shape of sample from real dataset

    Returns generator loss, and the image generated from noise (to be passed
    into the PatchNCE network).
    """
    solver.zero_grad()  # reset gradients

    # generate fake data
    noise = randn(real_data_shape)
    fake_data = G(noise)  # here, we DO want loss to backprop

    # train on fake data only
    prediction_fake = D(fake_data)
    loss = loss_G(prediction_fake)

    loss.backward()
    solver.step()

    return loss, fake_data


def loss_D(prediction_real, prediction_fake):
    """
    Computes the least squares GAN loss for the discriminator.

    See 3.2 of Mao et al.

    Inputs:
    - [torch.Tensor] prediction_real: discriminator's output given real data
    - [torch.Tensor] prediction_fake: discriminator's output given fake data

    Returns scalar least squares loss.
    """

    loss = mean((prediction_real - 1)**2) + mean(prediction_fake**2)
    return 0.5 * loss


def loss_G(prediction_fake):
    """
    Computes the least squares GAN loss for the generator. Note that the loss in
    this case does not depend on the discriminator's output given real data.

    See 3.2 of Mao et al.

    Inputs:
    - [torch.Tensor] prediction_fake: discriminator's output given fake data

    Returns scalar least squares loss.
    """
    loss = mean((prediction_fake - 1)**2)
    return 0.5 * loss
