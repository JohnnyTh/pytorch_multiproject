import copy
import itertools
import torch
import torch.nn as nn
import torch.optim as optim


class CycleGAN(nn.Module):

    def __init__(self, generator, discriminator, gan_loss, cycle_loss, identity_loss, hyperparams):
        super().__init__()
        self.ab_generator = generator
        self.ba_generator = copy.deepcopy(generator)
        self.ab_discriminator = discriminator
        self.ba_discriminator = copy.deepcopy(discriminator)
        self.criterionGAN = gan_loss
        self.criterionCycle = cycle_loss
        self.criterionIdt = identity_loss
        self._hyperparams = hyperparams

    def forward(self, real_a, real_b, step_flag,
                fake_b_disc=None, fake_a_disc=None):
        if step_flag == 'gen_step':
            device = real_a.device
            # discriminators require no gradients while optimizing generators
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], False)

            fake_b = self.ab_generator(real_a)
            rec_a = self.ba_generator(self.fake_b)
            fake_a = self.ba_generator(real_b)
            rec_b = self.ab_generator(self.fake_a)

            return fake_b, fake_a, rec_a, rec_b, self._loss_generators(real_a, real_b,
                                                                       fake_b, fake_a, rec_a, rec_b, device)

        elif step_flag == 'disc_step':
            device = real_a.device
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], True)
            return self._loss_discriminators(real_a, real_b, fake_b_disc, fake_a_disc, device)
        else:
            raise Exception('correct step flag name not provided !')

    def _loss_generators(self, real_a, real_b, fake_b, fake_a, rec_a, rec_b, device):
        lambda_idt = self._hyperparams['lambda_identity']
        lambda_a = self._hyperparams['lambda_a']
        lambda_b = self._hyperparams['lambda_b']

        # calculate identity loss
        idt_a = self.ab_generator(real_b)
        loss_idt_a = self.criterionIdt(idt_a, real_b) * lambda_b * lambda_idt

        idt_b = self.ba_generator(real_a)
        loss_idt_b = self.criterionIdt(idt_b, real_a) * lambda_a * lambda_idt

        # GAN loss ab_disc(ab_gen(A))
        prediction_b = self.ab_discriminator(fake_b)
        all_true_labels_b = torch.tensor([1.0]).expand_as(prediction_b).to(device)
        loss_ab_gen = self.criterionGAN(prediction_b, all_true_labels_b)
        # GAN loss ba_disc(ba_gen(B))
        prediction_a = self.ba_discriminator(fake_a)
        all_true_labels_a = torch.tensor([1.0]).expand_as(prediction_a).to(device)
        loss_ba_gen = self.criterionGAN(prediction_a, all_true_labels_a)

        # forward cycle loss
        loss_cycle_a = self.criterionCycle(rec_a, real_a) * lambda_a
        # backward cycle loss
        loss_cycle_b = self.criterionCycle(rec_b, real_b) * lambda_b
        # total loss
        loss_generators = loss_idt_a + loss_idt_b + loss_ab_gen + loss_ba_gen + loss_cycle_a + loss_cycle_b
        return loss_generators

    def _loss_discriminators(self, real_a, real_b, fake_b_disc, fake_a_disc, device):
        # take note that image pooling for fake(generated) images can be implemented here
        # calculate loss for ab_discriminator
        ab_disc_loss = self._loss_discriminators_base(self.ab_discriminator, real_b, fake_b_disc, device)

        # calculate loss for ba_discriminator
        ba_disc_loss = self._loss_discriminators_base(self.ba_discriminator, real_a, fake_a_disc, device)

        return ab_disc_loss, ba_disc_loss

    def _loss_discriminators_base(self, discrim, real, fake, device):

        pred_real = discrim(real)
        all_true_labels = torch.tensor([1.0]).expand_as(pred_real).to(device)
        loss_real = self.criterionGAN(pred_real, all_true_labels)

        pred_fake = discrim(fake)
        all_false_labels = torch.tensor([0.0]).expand_as(pred_fake).to(device)
        loss_fake = self.criterionGAN(pred_fake, all_false_labels)

        loss_discrim = (loss_real + loss_fake) / 2
        return loss_discrim

    def get_optims(self, lr=0.0002):
        optim_gen = optim.Adam(itertools.chain(self.ab_generator, self.ba_generator), lr=lr, betas=(0.5, 0.999))
        optim_disc = optim.Adam(itertools.chain(self.ab_discriminator, self.ba_discriminator), lr=lr, betas=(0.5, 0.999))
        return optim_gen, optim_disc

    @staticmethod
    def _set_requires_grad(models, requires_grad=False):
        for model in models:
            for param in model.parameters():
                param.requires_gard = requires_grad
