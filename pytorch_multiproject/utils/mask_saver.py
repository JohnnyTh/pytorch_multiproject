from PIL import Image
import numpy as np
from random import randrange


class MaskSaver:

    def __init__(self, save_dir, buffer_size=10, ):
        self.data = []
        self.save_dir = save_dir
        self.bufffer_size = buffer_size

    def accumulate(self, image, mask):
        if len(self.data) < self.bufffer_size:
            self.data.append([image, mask])

        else:
            # when the buffer is full, replace random element inside with chance 30%
            if randrange(1, 100) < 30:
                position = randrange(self.bufffer_size)
                self.data[position] = [image, mask]

    def generate_masked_img(self, epoch, mask_draw_precision=0.4, opacity=0.4):
        for idx, data in enumerate(self.data):
            image, masks = data
            image_prep = Image.fromarray(image)
            # add alpha channel to the original image
            image_prep.putalpha(255)
            for mask in masks:
                colors = self.generate_color_scheme()
                # firstly generate 3 color channels and alpha channel
                mask = np.repeat(mask, 4, axis=0)
                # replace ones at each color channel with respective color if mask probability > mask_draw_precision
                # and zero out the values below mask_draw_precision
                for color in range(len(colors)):
                    mask[color][mask[color] > int(255*mask_draw_precision)] = colors[color]
                    mask[color][mask[color] < int(255 * mask_draw_precision)] = 0
                # fill alpha channel values using R channel as a reference
                mask[3][mask[0] > 0] = int(255*opacity)
                mask[3][mask[0] < 0] = 0

                # convert the mask into H, W, C format
                mask = np.transpose(mask, (1, 2, 0))
                # convert the prepared mask into PIL Image object
                mask_prep = Image.fromarray(mask)
                # combine the mask and the image
                image_prep = Image.alpha_composite(image_prep, mask_prep)
            image_prep.save('Test_img_mask_{}_{}'.format(epoch, idx))

    @staticmethod
    def generate_color_scheme():
        return np.random.choice(range(256), size=3)
