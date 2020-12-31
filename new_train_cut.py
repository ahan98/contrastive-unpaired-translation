import torch
import bbml.training as training
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

    # Training Session
    training_context = training.TrainingSessionContext("patchNCEGAN", training_data, device, epochs, 20)

    return training.TrainingSession(training_context, [
        DiscriminatorOptimizationTask(generator, discriminator, discriminator_optimizer),
        GeneratorOptimizationTask(discriminator, generator, patchNCE, generator_optimizer, patchNCE_optimizer)
    ])
