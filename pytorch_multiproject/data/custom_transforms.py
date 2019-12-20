import torch
import random
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


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


class Compose(object):
    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, image, target=None):
        """Applies a sequence of transforms to an image and to a dict (optional) with bounding boxes, masks etc.

        Parameters
        ----------
        image (Tensor) - Tensor image of shape (C, H, W)
        target (optional, dict) - contains coordinates of bounding boxes, mask tensors etc. that need to be transformed
               alongside with the image.

        Returns
        -------
        image, target(optional) - transformed image and dict with masks, bounding boxes, etc.
        """
        if target is not None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

        else:
            for t in self.transforms:
                image = t(image)
            return image


class ResizeBboxImg(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        Parameters
        ----------
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            "PIL.Image.BILINEAR"
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target):
        """
        Parameters
        ----------
        img (Tensor) - Image Tensor  of size (C, H, W)
        target (dict) - contains masks and coordinates of bounding boxes that need to be resized

        Returns
        -------
        img, target - resized image and its bboxes, masks.
        """
        # get height and width of image before resizing
        w_old, h_old = img.size
        img = F.resize(img, self.size, self.interpolation)
        # get height and width of image after resizing
        w_new, h_new = img.size

        # get scale factors for horizontal and vertical dimensions
        w_scale = w_new/w_old
        h_scale = h_new/h_old
        # note that single bbox format is [x0, y0, x1, y1]
        bbox = target['boxes']
        # multiply x0, x1 by horizontal scale factor
        bbox[:, 0:3:2] = bbox[:, 0:3:2] * w_scale
        # multiply y0, y1 by vertical scale factor
        bbox[:, 1:4:2] = bbox[:, 1:4:2] * h_scale
        target['boxes'] = bbox

        if 'masks' in target:
            masks = target['masks']
            if len(masks.shape) == 3:

                transformed_masks = []
                for mask in masks:
                    mask = F.to_pil_image(mask.mul(255))
                    mask = F.resize(mask, self.size, self.interpolation)
                    mask = F.to_tensor(mask)
                    mask = mask.to(dtype=torch.uint8)
                    transformed_masks.append(mask)

                transformed_masks = torch.stack(transformed_masks)
                target['masks'] = transformed_masks
            else:
                raise ValueError('Please provide masks of correct shape: N, H, W')
        return img, target


# code below taken from https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
# end of copied code from https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
