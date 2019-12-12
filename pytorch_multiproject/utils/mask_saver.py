import os
import numpy as np
from PIL import Image
from random import randrange


class MaskSaver:

    def __init__(self, save_dir, buffer_size=10):
        self.data = []
        self.save_dir = save_dir
        self.buffer_size = buffer_size

    def accumulate(self, image, mask):
        if len(self.data) < self.buffer_size:
            self.data.append([image, mask])
        else:
            pass

    def generate_masked_img(self, epoch, selected_boxes_ind, mask_draw_precision=0.4, opacity=0.4):
        for idx, data in enumerate(self.data):
            image, masks = data
            idx_group = selected_boxes_ind[idx]
            image_prep = Image.fromarray(image)
            # add alpha channel to the original image
            image_prep.putalpha(255)

            # pick only those masks that correspond to the bounding boxes after non-max suppression
            if idx_group.dtype != int:
                idx_group = idx_group.astype(int)
            masks = masks[idx_group]

            for mask in masks:
                colors = self.generate_color_scheme()
                # firstly generate 3 color channels and alpha channel
                mask = np.repeat(mask, 4, axis=0)
                # replace ones at each color channel with respective color if mask probability > mask_draw_precision
                # and zero out the values below mask_draw_precision
                for channel in range(len(colors)):
                    bool_mask_keep = mask[channel] >= int(254*mask_draw_precision)
                    bool_mask_erase = mask[channel] < int(254*mask_draw_precision)

                    mask[channel][bool_mask_keep] = colors[channel]
                    mask[channel][bool_mask_erase] = 0
                # fill alpha channel values using R channel as a reference
                mask[3, :, :][mask[0, :, :] > 0] = int(255*opacity)
                mask[3, :, :][mask[0, :, :] == 0] = 0

                # convert the mask into H, W, C format
                mask = np.transpose(mask, (1, 2, 0))
                # convert the prepared mask into PIL Image object
                mask_prep = Image.fromarray(mask)
                # combine the mask and the image
                image_prep = Image.alpha_composite(image_prep, mask_prep)
            save_addr = os.path.join(self.save_dir, 'Test_img_mask_{}_{}'.format(epoch, idx))
            image_prep.save(save_addr, 'PNG')

    @staticmethod
    def generate_color_scheme():
        return np.random.choice(range(255), size=3)
