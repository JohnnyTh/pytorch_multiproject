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

# code below taken from https://github.com/pytorch/vision/blob/master/references/detection/transforms.py


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transform):
        self.transforms = transform

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ResizeBboxImg(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target):
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
                    mask = F.to_pil_image(mask)
                    mask = F.resize(mask, self.size, self.interpolation)
                    mask = F.to_tensor(mask)
                    transformed_masks.append(mask)

                transformed_masks = torch.stack(transformed_masks)
                target['masks'] = transformed_masks
            else:
                raise Exception('Please provide masks of correct shape: N, H, W')
        return img, target


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
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# end of copied code from https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
