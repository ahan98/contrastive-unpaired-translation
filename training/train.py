import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.GAN.Discriminator import Discriminator
from models.GAN.Encoder import Encoder
from models.GAN.Decoder import Decoder
from models.GAN.Generator import Generator
from models.PatchNCE.PatchNCE import PatchNCE
from data_utils import random_data_loader
from .trainers.GANTrainer import GANTrainer
from .trainers.PatchNCETrainer import PatchNCETrainer
from .trainers.utils import make_noise

# TODO: write logger


def train(X_dataset, Y_dataset, device, n_epochs=400, n_steps_D=1, lr=2e-3, print_every=100):

    # init networks
    D = Discriminator()

    G_enc, G_dec = Encoder(), Decoder()
    G = Generator(encoder=G_enc, decoder=G_dec)

    P = PatchNCE(G_enc)

    # init solvers
    solver_D = optim.Adam(D.parameters(), lr=lr)
    solver_G = optim.Adam(G.parameters(), lr=lr)
    solver_P = optim.Adam(P.parameters(), lr=lr)

    data_loader_Y = random_data_loader(Y_dataset)
    Y_iter = iter(data_loader_Y)

    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch, n_epochs))

        for n_batch, real_X in enumerate(X_dataset):

            # training discriminator
            for _ in range(n_steps_D):
                loss_D, pred_real_D, pred_fake_D = GANTrainer.train_discriminator(D, solver_D, real_X, device)

            # training generator
            loss_G, fake_X = GANTrainer.train_generator(G, D, solver_G, real_X.shape, device)

            # training PatchNCE
            loss_P = PatchNCETrainer.train_patchnce(P, solver_P, real_X, fake_X)

            # get random sample from Y, treating it as the "real" data
            try:
                real_Y = next(Y_iter)
            except StopIteration:
                Y_iter = iter(data_loader_Y)
                real_Y = next(Y_iter)

            real_Y = random_sampler_Y[epoch]
            noise = make_noise(real_Y.shape)
            fake_Y = G(noise)
            loss_P += PatchNCETrainer.train_patchnce(P, solver_P, real_Y, fake_Y)

        if n_batch % print_every == 0:
            print("loss_D: {:e}, loss_G: {:e}, loss_P: {:e}"
                  .format(loss_D, loss_G, loss_P))


# if __name__ == "__main__":
    # hyperparameters
    # ...
    # training(...)
