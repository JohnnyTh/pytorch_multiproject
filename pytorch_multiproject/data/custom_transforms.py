import math
import warnings
import torch
import random
from PIL import Image
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

    def __call__(self, image, target):
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
        w_old, h_old = image.size
        image = F.resize(image, self.size, self.interpolation)
        # get height and width of image after resizing
        w_new, h_new = image.size

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
                    mask = F.to_tensor(mask).squeeze()
                    mask = mask.to(dtype=torch.uint8)
                    transformed_masks.append(mask)

                transformed_masks = torch.stack(transformed_masks)
                target['masks'] = transformed_masks
            else:
                raise ValueError('Please provide masks of correct shape: N, H, W')
        return image, target


class RandomResizedCropBbox:
    """ Based on https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomCrop
    Crop the given PIL Image to random size and aspect ratio.
    Additionally corrects the size of the bounding boxes and masks

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
    
    @staticmethod
    def get_params(image, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            image (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = image.size
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                y = random.randint(0, height - h)
                x = random.randint(0, width - w)
                return y, x, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        y = (height - h) // 2
        x = (width - w) // 2
        return y, x, h, w
    
    def __call__(self, image, target):
        """
        Args:
            image (PIL Image): Image to be cropped and resized.
            target (dict): contains masks and coordinates of bounding boxes to be cropped and resized

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        y, x, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, y, x, h, w, self.size, self.interpolation)
        # get height and width of image after resizing
        w_new, h_new = image.size

        # note that single bbox format is [x0, y0, x1, y1]
        bbox = target['boxes']
        # create mask that will drop all bounding boxes that have no intersection with cropped region
        # the intersection conditions are x0 < x + w, y0 < y + h, x1 > x, y1 > y
        drop_mask = (bbox[:, 0] < x + w) & (bbox[:, 1] < y + h) & (bbox[:, 2] > x) & (bbox[:, 3] > y)
        bbox = bbox[drop_mask]
        # move bonding box coordinates to coordinate system of cropped region
        bbox[:, 0:3:2] = bbox[:, 0:3:2] - x
        bbox[:, 1:4:2] = bbox[:, 1:4:2] - y
        # clamp the bbox coordinate values at the boundaries of cropped region
        bbox[:, 0:2][bbox[:, 0:2] < 0] = 0
        bbox[:, 3][bbox[:, 3] > w] = w
        bbox[:, 4][bbox[:, 4] > h] = h
        # get scale factors for horizontal and vertical dimensions
        w_scale = w_new/w
        h_scale = h_new/h
        # adjust x0, x1 by horizontal scale factor
        bbox[:, 0:3:2] = bbox[:, 0:3:2] * w_scale
        # adjust y0, y1 by vertical scale factor
        bbox[:, 1:4:2] = bbox[:, 1:4:2] * h_scale
        target['boxes'] = bbox

        if 'masks' in target:
            masks = target['masks']
            if len(masks.shape) == 3:

                transformed_masks = []
                for mask in masks:
                    mask = F.to_pil_image(mask.mul(255))
                    mask = F.resized_crop(mask, y, x, h, w, self.size, self.interpolation)
                    mask = F.to_tensor(mask).squeeze()
                    mask = mask.to(dtype=torch.uint8)
                    transformed_masks.append(mask)

                transformed_masks = torch.stack(transformed_masks)
                target['masks'] = transformed_masks
            else:
                raise ValueError('Please provide masks of correct shape: N, H, W')
        return image, target


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
