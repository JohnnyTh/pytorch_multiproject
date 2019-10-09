import json
import torch
from pathlib import Path
import torch.nn as nn
from collections import OrderedDict

def weights_init(m):
    """Xavier weight initialization func."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

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