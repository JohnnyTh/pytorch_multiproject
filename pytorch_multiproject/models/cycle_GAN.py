import copy
import torch
import torch.nn as nn

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.
        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class CycleGAN(nn.Module):

    def __init__(self, generator, discriminator, hyperparams, device):
        super().__init__()
        self.device = device
        self.ab_generator = generator
        self.ba_generator = copy.deepcopy(generator)
        self.ab_discriminator = discriminator
        self.ba_discriminator = copy.deepcopy(discriminator)
        self.criterionGAN = GANLoss('lsgan').to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self._hyperparams = hyperparams
        self.fake_B = None
        self.fake_A = None
        self.rec_A = None
        self.rec_B = None

    def forward(self, real_A, real_B, step_flag):
        if step_flag == 'gen_step':
            # discriminators require no gradients while optimizing generators
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], False)
            self.fake_B = self.ab_generator(real_A)
            self.rec_A = self.ba_generator(self.fake_B)
            self.fake_A = self.ba_generator(real_B)
            self.rec_B = self.ab_generator(self.fake_A)

            return self.fake_B, self.fake_A, self.rec_A, self.rec_B, self._loss_generators(real_A, real_B)

        elif step_flag == 'disc_step':
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], True)
            return self._loss_discriminators()
        else:
            raise Exception('correct step flag name not provided !')

    def _loss_generators(self, real_A, real_B):
        lambda_idt = self._hyperparams['lambda_identity']
        lambda_A = self._hyperparams['lambda_A']
        lambda_B = self._hyperparams['lambda_B']

        # calculate identity loss
        idt_A = self.ab_generator(real_B)
        loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_B * lambda_idt

        idt_B = self.ba_generator(real_A)
        loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_A * lambda_idt

        # GAN loss ab_disc(ab_gen(A))
        loss_ab_gen = self.criterionGAN(self.ab_discriminator(self.fake_B), True)
        # GAN loss ba_disc(ba_gen(B))
        loss_ba_gen = self.criterionGAN(self.ba_discriminator(self.fake_A), True)

        # forward cycle loss
        loss_cycle_A = self.criterionCycle(self.rec_A, real_A) * lambda_A
        # backward cycle loss
        loss_cycle_B = self.criterionCycle(self.rec_B, real_B) * lambda_B
        # total loss
        loss_genertators = loss_idt_A + loss_idt_B + loss_ab_gen + loss_ba_gen + loss_cycle_A + loss_cycle_B
        return loss_genertators

    def _loss_discriminators(self):
        pass

    @staticmethod
    def _set_requires_grad(models, requires_grad=False):
        for model in models:
            for param in model.parameters():
                param.requires_gard = requires_grad