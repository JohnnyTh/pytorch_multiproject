import torch


class Denormalize(object):
    """Denormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    Returns:
        Tensor: Denormalized Tensor image.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        if self.mean is None:
            self.mean = [0.5, 0.5, 0.5]

        self.std = std
        if self.std is None:
            self.std = [0.5, 0.5, 0.5]

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor = tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

        return tensor
