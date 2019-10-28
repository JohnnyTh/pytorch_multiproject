import copy
import torch
import torch.nn as nn


class CycleGAN(nn.Module):

    def __init__(self, generator, discriminator):
        super().__init__()
        self.ab_generator = generator
        self.ba_generator = copy.deepcopy(generator)
        self.ab_discriminator = discriminator
        self.ba_discriminator = copy.deepcopy(discriminator)
        self.fake_B = None
        self.fake_A = None
        self.rec_A = None
        self.rec_B = None

    def forward(self, input_, target, step_flag):
        if step_flag == 'gen_step':
            # discriminators require no gradients while optimizing generators
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], False)
            self.fake_B = self.ab_generator(input_)
            self.rec_A = self.ba_generator(self.fake_B)
            self.fake_A = self.ba_generator(target)
            self.rec_B = self.ab_generator(self.fake_A)

            return self.fake_B, self.fake_A, self.rec_A, self.rec_B, self._loss_generators()

        elif step_flag == 'disc_step':
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], True)
            return self._loss_discriminators()
        else:
            raise Exception('correct step flag name not provided !')

    @staticmethod
    def _set_requires_grad(models, requires_grad=False):
        for model in models:
            for param in model.parameters():
                param.requires_gard = requires_grad

    def _loss_generators(self):
        pass

    def _loss_discriminators(self):
        pass
