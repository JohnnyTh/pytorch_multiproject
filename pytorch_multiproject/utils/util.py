import mock
import torch
import torch.nn as nn
from torch.nn import init


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


def normal_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

        init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def apply_weight_init(model, mode, init_gain=0.02):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if mode == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif mode == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif mode == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif mode == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


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


def cat_list(images, fill_value=0):
    # source https://github.com/pytorch/vision/blob/master/references/segmentation/utils.py
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    # source https://github.com/pytorch/vision/blob/master/references/segmentation/utils.py
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets
