import torch
import torch.nn as nn

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

