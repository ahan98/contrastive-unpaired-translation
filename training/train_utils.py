"""
A collection of helper methods for training
"""

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from collections import defaultdict
from models.GAN.Discriminator import Discriminator
from models.GAN.Generator import Generator
from models.PatchNCE.PatchNCE import PatchNCE
from glob import glob

def make_dataloader(batches_by_class, class_name, image_size=256,
                    minibatch_size=1, replacement=False):
    """
    Returns a Dataloader from the batch of tensor images stored in
    batches_by_class[class_name].

    Inputs:
    - [dict] batches_by_class: dictionary of class names to batch
    - [String] class_name: class name of desired batch
    - [int] image_size: dimension to scale images
    - [int] minibatch_size: number of samples to draw from X and Y
    - [bool] replacement: if True, create DataLoader with a RandomSampler that
      samples with replacement
    """

    # get batch for path/class_name
    batch = batches_by_class[class_name]

    # condense batches into (N+1)-dimensional tensors
    batch = torch.stack(batch)

    # convert to dataloader
    if replacement:
        sampler = RandomSampler(batch, replacement=True)
        dataloader = DataLoader(batch, batch_size=minibatch_size, sampler=sampler)
    else:
        dataloader = DataLoader(batch, shuffle=True, batch_size=minibatch_size)

    return dataloader


def get_batches_from_path(path, image_size=256):
    """
    Returns a dictionary mapping each subdirectory to a batch of its images.

    Inputs:
    - [String] path: path containing subdirectories of images
    - [int] image_size: dimension to scale images
    """

    # preprocess images to (image_size x image_size), then convert to tensors
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),  # makes images square
        T.ToTensor()
    ])

    image_folder = ImageFolder(path, transform=transform)
    class_names = image_folder.classes
    batches_by_class = defaultdict(list)

    for image_PIL, class_index in image_folder:
        class_name = class_names[class_index]
        batches_by_class[class_name].append(image_PIL)

    return batches_by_class


def load_models_and_losses(path_to_checkpoints, lr_D, lr_P, lr_G, device="cpu"):
    # init models
    D = Discriminator().to(device)
    G = Generator().to(device)
    P = PatchNCE(G.encoder).to(device)

    # init Adam optimizers
    solver_D = torch.optim.Adam(D.parameters(), lr=lr_D)
    solver_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    solver_P = torch.optim.Adam(P.parameters(), lr=lr_P)

    # init Adam optimizers
    scheduler_D = torch.optim.lr_scheduler.StepLR(solver_D, step_size=20, gamma=0.1)
    scheduler_G = torch.optim.lr_scheduler.StepLR(solver_G, step_size=20, gamma=0.1)
    scheduler_P = torch.optim.lr_scheduler.StepLR(solver_P, step_size=20, gamma=0.1)

    # load losses from checkpoint
    loss_checkpoint = glob(path_to_checkpoints + "minibatch_losses*.pt")
    if loss_checkpoint:
        print("Loading checkpoint for losses...")
        loss_per_minibatch = torch.load(loss_checkpoint[0])
    else:
        loss_per_minibatch = {"discriminator": [], "generator": [], "patchNCE": []}

    # load state dicts for models and solvers from checkpoints
    model_checkpoints = glob(path_to_checkpoints + "checkpoint*.pt")
    for checkpoint in model_checkpoints:
        state = torch.load(checkpoint)
        model_name = state["model_name"]
        print("Loading checkpoint for {}...".format(model_name))

        model_state = state["model_state_dict"]
        solver_state = state["solver_state_dict"]
        # scheduler_state = state["scheduler_state_dict"]

        if model_name == "discriminator":
            D.load_state_dict(model_state)
            solver_D.load_state_dict(solver_state)
            # scheduler_D.load_state_dict(scheduler_state)

        elif model_name == "generator":
            G.load_state_dict(model_state)
            solver_G.load_state_dict(solver_state)
            # scheduler_G.load_state_dict(scheduler_state)

        elif model_name == "patchNCE":
            P.load_state_dict(model_state)
            solver_P.load_state_dict(solver_state)
            # scheduler_P.load_state_dict(scheduler_state)

    models_dict = {
        "discriminator": (D, solver_D, scheduler_D),
        "generator": (G, solver_G, scheduler_G),
        "patchNCE": (P, solver_P, scheduler_P),
    }

    return models_dict, loss_per_minibatch

