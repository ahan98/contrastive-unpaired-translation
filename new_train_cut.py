import torch
import bbml.models.training as training
from .models.GAN import *
from .models.PatchNCE import *
from .optimization.GeneratorOptimizationTask import *
from .optimization.DiscriminatorOptimizationTask import *

def make_PatchNCEGAN_training_session(training_data: bbml.SafeTupleIterator,
                                      epochs: int,
                                      device: torch.device,
                                      generator_learning_rate: float = 2e-3,
                                      discriminator_learning_rate: float = 2e-3,
                                      patchNCE_learning_rate: float = 2e-3) -> training.TrainingSession:

    # print_every = args.print if args.print else len(X_train_dataloader)
    # checkpoint_epoch = args.save if args.save else 0
    training_context = training.TrainingSessionContext(training_data, epochs, device)

    # Models
    generator = Generator(Encoder(), Decoder()).to(device)
    generator_optimizer = training.ModelOptimizer("generator.optimizer",
                                                  torch.optim.Adam(generator.parameters(), lr=generator_learning_rate))

    patchNCE = PatchNCE().to(device)
    patchNCE_optimizer = training.ModelOptimizer("patchNCE.optimizer",
                                                 torch.optim.Adam(patchNCE.parameters(), lr=patchNCE_learning_rate))

    discriminator = Discriminator().to(device)
    discriminator_optimizer = training.ModelOptimizer("discriminator.optimizer",
                                                      torch.optim.Adam(discriminator.parameters(),
                                                                       lr=discriminator_learning_rate))

    return training.TrainingSession(training_context, [
        DiscriminatorOptimizationTask(generator, discriminator, discriminator_optimizer),
        GeneratorOptimizationTask(discriminator, generator, patchNCE, generator_optimizer, patchNCE_optimizer)
    ])

    # models_dict, loss_per_minibatch = \
    #     train(models_dict, loss_per_minibatch, X_train_dataloader,
    #           Y_train_dataloader, device=device, n_epochs=n_epochs,
    #           print_every=print_every, checkpoint_epoch=checkpoint_epoch)

    # # Save final states after training
    # print("Training completed.")
    # save_models(models_dict, n_epochs)
    # save_losses(loss_per_minibatch, n_epochs)
