import torch
import torch.nn as nn
import mock


def same_padding_calc(inp_shape, kernel_shape, stride):
    """
       !Attention - only square image padding calculation implemented!
       Calculates the size of 'same' padding  for CONV layers.

       Args:
       kernel_shape (int or tuple): the shape of the kernel(filter).
       inp_shape (int or tuple): the shape of the input.
       stride (int).

       Returns:
       res (int or tuple): 'same' padding size.
    """
    if type(inp_shape) == int and type(kernel_shape) == int:
        res = (inp_shape * stride - inp_shape - stride + kernel_shape) // 2
        return res
    elif type(inp_shape) == tuple and type(kernel_shape) == int:
        res = None
        return res
    elif type(inp_shape) == int and type(kernel_shape) == tuple:
        res = None
        return res
    elif type(inp_shape) == tuple and type(kernel_shape) == tuple:
        res = None
        return res
    else:
        res = None
        return res


def freeze_unfreeze_model(model, mode):
    """A simple function to freeze or unfreeze the model weights"""
    if mode == 'freeze':
        for param in model.parameters():
            param.requires_grad = False
    if mode == 'unfreeze':
        for param in model.parameters():
            param.requires_grad = True


def weights_inint_seq(sequential):
    # A loop to iterate over the modules in nn.Sequential
    for module in sequential:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)


class MockGeneratorFoo:

    def __init__(self):
        """
        Imitates work of a generator in GAN.
        Returns a tensor with random values in range 0-1 in the shape (64, 3, 32, 32),
        imitating batch of 64 32x32 images with 3 channels.
        """
        pass

    def __call__(self, real_img):
        device = real_img.device
        mock_fake_img = torch.rand((64, 3, 32, 32))
        return mock_fake_img.to(device)

    @staticmethod
    def parameters():
        return [mock.MagicMock()] * 5

    @staticmethod
    def to(*args):
        pass


class MockDiscriminatorFoo:
    """
       Imitates work of a discriminator in GAN.
       Returns a tensor with random values 0 or 1 in the shape (64, 1),
       imitating predictictions for batch of 64 images.
    """

    def __init__(self):
        pass

    def __call__(self, real_or_fake):
        device = real_or_fake.device
        mock_pred = torch.randint(2, (64, 1)).float()
        return mock_pred.to(device)

    @staticmethod
    def parameters():
        return [mock.MagicMock()] * 5

    @staticmethod
    def to(*args):
        pass
