import mock
import torch
import pickle
import torch.nn as nn
import torch.distributed as dist
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
                raise NotImplementedError('initialization method [%s] is not implemented' % mode)
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


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def collate_fn(batch):
    # source https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    # source https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def all_gather(data):
    """
    source https://github.com/pytorch/vision/blob/master/references/detection/utils.py

    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list