import os
import torch
import numpy as np
from PIL import Image
from data.generic_dataset import GenericDataset


class PennFudanDataset(GenericDataset):

    def __init__(self, root, *args, transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transforms = transforms

        self.imgs_root = os.path.basename(self._found_dataset[0]['root'])
        self.mask_root = os.path.basename(self._found_dataset[1]['root'])
        self.imgs = sorted(self._found_dataset[0]['names'])
        self.masks = sorted(self._found_dataset[1]['names'])

    def __getitem__(self, item):
        img_path = os.path.join(self.root, self.imgs_root, self.imgs[item])
        mask_path = os.path.join(self.root, self.mask_root, self.masks[item])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([item])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
