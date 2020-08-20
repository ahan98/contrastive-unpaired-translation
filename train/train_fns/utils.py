import torch

def real_data_target(shape, device):
    '''
    Tensor containing ones, with shape = size
    '''
    target = torch.ones(shape, device=device)
    return target


def fake_data_target(shape, device):
    '''
    Tensor containing zeros, with shape = size
    '''
    target = torch.zeros(shape, device=device)
    return target


def make_noise(shape):
    return torch.randn(shape)
