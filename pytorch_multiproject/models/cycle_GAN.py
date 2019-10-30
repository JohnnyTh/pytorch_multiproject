import copy
import torch
import torch.nn as nn


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

    def forward(self, real_A, real_B, step_flag,
                fake_B_disc=None, fake_A_disc=None):
        if step_flag == 'gen_step':
            device = real_A.device
            # discriminators require no gradients while optimizing generators
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], False)

            fake_B = self.ab_generator(real_A)
            rec_A = self.ba_generator(self.fake_B)
            fake_A = self.ba_generator(real_B)
            rec_B = self.ab_generator(self.fake_A)

            return fake_B, fake_A, rec_A, rec_B, self._loss_generators(real_A, real_B,
                                                                       fake_B, fake_A, rec_A, rec_B, device)

        elif step_flag == 'disc_step':
            device = real_A.device
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], True)
            return self._loss_discriminators(real_A, real_B, fake_B_disc, fake_A_disc, device)
        else:
            raise Exception('correct step flag name not provided !')

    def _loss_generators(self, real_A, real_B, fake_B, fake_A, rec_A, rec_B, device):
        lambda_idt = self._hyperparams['lambda_identity']
        lambda_A = self._hyperparams['lambda_A']
        lambda_B = self._hyperparams['lambda_B']

        # calculate identity loss
        idt_A = self.ab_generator(real_B)
        loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_B * lambda_idt

        idt_B = self.ba_generator(real_A)
        loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_A * lambda_idt

        # GAN loss ab_disc(ab_gen(A))
        prediction_B = self.ab_discriminator(fake_B)
        all_true_labels_B = torch.tensor([1.0]).expand_as(prediction_B).to(device)
        loss_ab_gen = self.criterionGAN(prediction_B, all_true_labels_B)
        # GAN loss ba_disc(ba_gen(B))
        prediction_A = self.ba_discriminator(fake_A)
        all_true_labels_A = torch.tensor([1.0]).expand_as(prediction_A).to(device)
        loss_ba_gen = self.criterionGAN(prediction_A, all_true_labels_A)

        # forward cycle loss
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A
        # backward cycle loss
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B
        # total loss
        loss_generators = loss_idt_A + loss_idt_B + loss_ab_gen + loss_ba_gen + loss_cycle_A + loss_cycle_B
        return loss_generators

    def _loss_discriminators(self, real_A, real_B, fake_B_disc, fake_A_disc, device):
        # take note that image pooling for fake(generated) images can be implemented here
        # calculate loss for ab_discriminator
        ab_disc_loss = self._loss_discriminators_base(self.ab_discriminator, real_B, fake_B_disc, device)

        # calculate loss for ba_discriminator
        ba_disc_loss = self._loss_discriminators_base(self.ba_discriminator, real_A, fake_A_disc, device)

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

    @staticmethod
    def _set_requires_grad(models, requires_grad=False):
        for model in models:
            for param in model.parameters():
                param.requires_gard = requires_grad
