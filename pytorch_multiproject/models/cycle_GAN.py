import copy
import itertools
import torch
import torch.nn as nn
import torch.optim as optim


class GanOptimizer:
    """
       Implements optimizer behaviour for CycleGAN, which uses two optimizers
       (one for generators and one for discriminators) instead of one.
    """
    def __init__(self, optim_generator, optim_discriminator):
        self.generator_optim = optim_generator
        self.discriminator_optim = optim_discriminator

    def __str__(self):
        # string representation for logging
        gen = str(self.generator_optim)
        disc = str(self.discriminator_optim)
        out = '\nGenerator optim state: {}\nDiscriminator optim state: {}'.format(gen, disc)
        return out

    def zero_grad(self, optim_):
        # zeroes out the gradients for generators or discriminators
        if optim_ == 'optim_gen':
            self.generator_optim.zero_grad()
        elif optim_ == 'optim_disc':
            self.discriminator_optim.zero_grad()
        else:
            raise ValueError('Provide correct optim name (optim_gen or optim_disc)')

    def step(self, optim_):
        # updates the model's parameters for generators or discriminators
        if optim_ == 'optim_gen':
            self.generator_optim.step()
        elif optim_ == 'optim_disc':
            self.discriminator_optim.step()
        else:
            raise ValueError('Provide correct optim name (optim_gen or optim_disc)')

    def state_dict(self):
        # get a dictionary with optim parameters for model serialization
        return {'optim_gen_state': self.generator_optim.state_dict(),
                'optim_disc_state': self.discriminator_optim.state_dict()}

    def load_state_dict(self, dict_):
        # restores the optimizers' parameters from a checkpoint
        self.generator_optim.load_state_dict(dict_['optim_gen_state'])
        self.discriminator_optim.load_state_dict(dict_['optim_disc_state'])

    def change_lr(self, lr):
        # changes the lr of both optimizers
        for param_group_1, param_group_2 in zip(self.generator_optim, self.discriminator_optim):
            param_group_1['lr'] = lr
            param_group_2['lr'] = lr


class GanLrScheduler:
    """
       Implements lr scheduler behaviour for CycleGAN, which uses two schedulers
       (one for generators and one for discriminators) instead of one.
    """
    def __init__(self, sched_gen, sched_disc):
        self.sched_gen = sched_gen
        self.sched_disc = sched_disc

    def __str__(self):
        # string representation for logging
        gen = str(self.sched_gen.state_dict())
        disc = str(self.sched_disc.state_dict())
        out = '\nGenerator lr_sched state: {}\nDiscriminator lr_sched state: {}'.format(gen, disc)
        return out

    def step(self, sched):
        # updates the optim's lr for generators or discriminators
        if sched == 'sched_gen':
            self.sched_gen.step()
        elif sched == 'sched_disc':
            self.sched_disc.step()
        else:
            raise ValueError('Provide correct lr sched name (sched_gen or sched_disc)')

    def state_dict(self):
        # get a dictionary with scheduler parameters for model serialization
        return {'sched_gen_state': self.sched_gen.state_dict(),
                'sched_disc_state': self.sched_disc.state_dict()}

    def load_state_dict(self, dict_):
        # restores the schedulers' parameters from a checkpoint
        self.sched_gen.load_state_dict(dict_['sched_gen_state'])
        self.sched_disc.load_state_dict(dict_['sched_disc_state'])


class ResBlock(nn.Module):
    """A single resblock for CycleGan generators"""
    def __init__(self, skip_relu):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=True),
                                   nn.InstanceNorm2d(256),
                                   nn.ReLU(True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=True),
                                   nn.InstanceNorm2d(256)
                                   )
        self.skip_relu = skip_relu
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = x + self.block(x)

        # the original implementation omitted ReLU after skip connection
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
        if not self.skip_relu:
            # missing relu added!
            out = self.relu(out)
        return out


class GanGenerator(nn.Module):
    """
    The generator  includes downsampling, transformation (resblocks), and upsampling layers.
    For 256x256 images the recommended number of resblocks is 9, for 128x128 - 6.
    """
    def __init__(self, num_resblocks=9, skip_relu=False):
        super(GanGenerator, self).__init__()

        block_initial = [nn.ReflectionPad2d(3),
                         nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=True),
                         nn.InstanceNorm2d(64),
                         nn.ReLU(True)]

        downsampling = [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(True),
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.InstanceNorm2d(256),
                        nn.ReLU(True)
                        ]

        resblocks = [ResBlock(skip_relu=skip_relu)] * num_resblocks

        upsampling = [nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                      nn.InstanceNorm2d(128),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(True)
                      ]

        block_last = [nn.ReflectionPad2d(3),
                      nn.Conv2d(64, 3, kernel_size=7, padding=0),
                      nn.Tanh()]

        pipeline = block_initial + downsampling + resblocks + upsampling + block_last
        self.model = nn.Sequential(*pipeline)

    def forward(self, input_):
        return self.model(input_)


class GanDiscriminator(nn.Module):
    """
        Discriminator with 70x70 receptive field on 256x256 images.
    """
    def __init__(self):
        super(GanDiscriminator, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(0.2, True),

                                   nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                   nn.InstanceNorm2d(128),
                                   nn.LeakyReLU(0.2, True),

                                   nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(0.2, True),

                                   nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
                                   nn.InstanceNorm2d(512),
                                   nn.LeakyReLU(0.2, True),

                                   nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
                                   )

    def forward(self, x):
        out = self.model(x)
        return out


class ResBlockSmall(nn.Module):
    """
        Smaller version of resblock layer for low-resolution (< 128x128) images
    """
    def __init__(self, skip_relu):
        super(ResBlockSmall, self).__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=0, bias=True),
                                   nn.InstanceNorm2d(128),
                                   nn.ReLU(True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(128, 128, kernel_size=3, padding=0, bias=True),
                                   nn.InstanceNorm2d(128)
                                   )
        self.skip_relu = skip_relu
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = x + self.block(x)

        if not self.skip_relu:
            # missing relu added!
            out = self.relu(out)
        return out


class GanGeneratorSmall(nn.Module):
    """
        Smaller version of generator for low-resolution (< 128x128) images
    """
    def __init__(self, num_resblocks=6, skip_relu=False):
        super(GanGeneratorSmall, self).__init__()

        block_initial = [nn.ReflectionPad2d(3),
                         nn.Conv2d(3, 32, kernel_size=7, padding=0, bias=True),
                         nn.InstanceNorm2d(32),
                         nn.ReLU(True)]

        downsampling = [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(True),
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(True)
                        ]

        resblocks = [ResBlockSmall(skip_relu=skip_relu)] * num_resblocks

        upsampling = [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(True),
                      nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                      nn.InstanceNorm2d(32),
                      nn.ReLU(True)
                      ]

        block_last = [nn.ReflectionPad2d(3),
                      nn.Conv2d(32, 3, kernel_size=7, padding=0),
                      nn.Tanh()]

        pipeline = block_initial + downsampling + resblocks + upsampling + block_last
        self.model = nn.Sequential(*pipeline)

    def forward(self, input_):
        return self.model(input_)


class GanDiscriminatorSmall(nn.Module):
    """
        Smaller version of discriminator for low-resolution (< 128x128) images
    """
    def __init__(self):
        super(GanDiscriminatorSmall, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(0.2, True),

                                   nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                   nn.InstanceNorm2d(64),
                                   nn.LeakyReLU(0.2, True),

                                   nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                   nn.InstanceNorm2d(128),
                                   nn.LeakyReLU(0.2, True),

                                   nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(0.2, True),

                                   nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
                                   )

    def forward(self, x):
        out = self.model(x)
        return out


class CycleGAN(nn.Module):

    def __init__(self, generator, discriminator, gan_loss, cycle_loss, identity_loss, hyperparams):
        """
        Implements CycleGan from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
        :param generator: callable, based on nn.Module, contains downsampling, transformation, and upsampling layers.
                          The role of generator is to apply a transformation to an input image so that the
                          image is translated to the domain of a set of target images.
        :param discriminator: callable, based on nn.Module, is composed of a sequence of convolution layers
                              with increasing number of filters and decreasing H and W. The role of
                              discriminator is to distinguish between fake images produced by generators
                              and the real images from target domain.
        :param gan_loss: callable, adversarial loss criteria. nn.MSELoss() recommended.
        :param cycle_loss: callable, cycle consistency loss criteria. nn.L1Loss() recommended.
        :param identity_loss: callable, identity mapping loss criteria. nn.L1Loss() recommended.
        :param hyperparams: dict, contains hyperparameters to modify the magnitude of loss function outputs
        """
        super().__init__()
        self.ab_generator = generator
        self.ba_generator = copy.deepcopy(generator)
        self.ab_discriminator = discriminator
        self.ba_discriminator = copy.deepcopy(discriminator)
        self.criterionGAN = gan_loss
        self.criterionCycle = cycle_loss
        self.criterionIdt = identity_loss
        self._hyperparams = hyperparams

    def forward(self, real_a, real_b, step_flag, fake_b_disc=None, fake_a_disc=None):
        """
        Implements forward pass trough generators (if step_flag == 'gen_step') or
        discriminators (if step_flag == 'disc_step')
        :param real_a:  a tensor of original image(s) from domain A.
        :param real_b:  a tensor of original image(s) from domain B.
        :param step_flag: str, determines whether forward pass is done through generators or discriminators.
        :param fake_b_disc: a tensor of fake images produced by ab generator.
        :param fake_a_disc: a tensor of fake images produced by ba generator.
        :return: fake_b: tensor of fake images produced by ab generator.
                 fake_a: tensor of fake images produced by ba generator.
                 rec_a: reconstructed image 'a' after -> ba_gen(ab_gen(a))
                 rec_b: reconstructed image 'b' after -> ab_gen(ba_gen(b))
                 self._loss_generators(): loss for generators, returned only if step_flag == 'gen_step'
                 self._loss_discriminators(): loss for discriminators, returned only if step_flag == 'disc_step'
        """
        if step_flag == 'gen_step':
            device = real_a.device
            fake_b = self.ab_generator(real_a)
            rec_a = self.ba_generator(fake_b)
            fake_a = self.ba_generator(real_b)
            rec_b = self.ab_generator(fake_a)

            # discriminators require no gradients while optimizing generators
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], False)

            return fake_b, fake_a, rec_a, rec_b, self._loss_generators(real_a, real_b,
                                                                       fake_b, fake_a, rec_a, rec_b, device)

        elif step_flag == 'disc_step':
            device = real_a.device
            self._set_requires_grad([self.ab_discriminator, self.ba_discriminator], True)
            return self._loss_discriminators(real_a, real_b, fake_b_disc, fake_a_disc, device)
        else:
            raise ValueError('correct step flag name not provided !')

    def _loss_generators(self, real_a, real_b, fake_b, fake_a, rec_a, rec_b, device):
        """
        Calculates the total generators loss as a sum of identity, adversarial, and cycle consistency losses.
        :param real_a: a tensor of original image(s) from domain A.
        :param real_b: a tensor of original image(s) from domain B.
        :param fake_b: tensor of fake images produced by ab generator.
        :param fake_a: tensor of fake images produced by ba generator.
        :param rec_a:  reconstructed image 'a' after -> ba_gen(ab_gen(a)).
        :param rec_b:  reconstructed image 'b' after -> ab_gen(ba_gen(b)).
        :param device: str, used device type 'cuda:0' if gpu is available else 'cpu'.
        :return: loss_generators: total loss of the generators.
        """
        lambda_idt = self._hyperparams.get('lambda_identity', 0.5)
        lambda_a = self._hyperparams.get('lambda_a', 10.0)
        lambda_b = self._hyperparams.get('lambda_b', 10.0)

        # identity loss
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
        """
        Calculates the loss of ab and ba discriminators.
        :param real_a: a tensor of original image(s) from domain A.
        :param real_b: a tensor of original image(s) from domain B.
        :param fake_b_disc: a tensor of fake images produced by ab generator.
        :param fake_a_disc: tensor of fake images produced by ba generator.
        :param device: str, used device type 'cuda:0' if gpu is available else 'cpu'.
        :return: ab_disc_loss: loss of ab discriminator
                 ba_disc_loss: loss of ba discriminator
        """
        # take note that image pooling for fake(generated) images can be implemented here
        # calculate loss for ab_discriminator
        ab_disc_loss = self._loss_discriminators_base(self.ab_discriminator, real_b, fake_b_disc, device)

        # calculate loss for ba_discriminator
        ba_disc_loss = self._loss_discriminators_base(self.ba_discriminator, real_a, fake_a_disc, device)

        return ab_disc_loss, ba_disc_loss

    def _loss_discriminators_base(self, discrim, real, fake, device):
        """
        Basic method to calculate discriminator loss for ab or ba discriminator.
        :param discrim: callable, ab or ba discriminator.
        :param real: a tensor of original image(s).
        :param fake: a tensor of fake image(s) produced from the original.
        :param device: str, used device type 'cuda:0' if gpu is available else 'cpu'.
        :return: loss_discrim: discriminator loss.
        """
        pred_real = discrim(real)
        all_true_labels = torch.tensor([1.0]).expand_as(pred_real).to(device)
        loss_real = self.criterionGAN(pred_real, all_true_labels)

        pred_fake = discrim(fake.detach())
        all_false_labels = torch.tensor([0.0]).expand_as(pred_fake).to(device)
        loss_fake = self.criterionGAN(pred_fake, all_false_labels)

        loss_discrim = (loss_real + loss_fake) / 2
        return loss_discrim

    def get_optims(self, lr=0.0002):
        """
        Returns instances of Adam optimizers for generators and discriminators.
        :param lr: float, learning rate.
        :return:optim_gen: generators optimizer.
                optim_disc: discriminators optimizer.
        """
        optim_gen = optim.Adam(itertools.chain(self.ab_generator.parameters(), self.ba_generator.parameters()),
                               lr=lr, betas=(0.5, 0.999))
        optim_disc = optim.Adam(itertools.chain(self.ab_discriminator.parameters(), self.ba_discriminator.parameters()),
                                lr=lr, betas=(0.5, 0.999))
        return optim_gen, optim_disc

    @staticmethod
    def _set_requires_grad(models, requires_grad=False):
        """
        Disables or enables gradient computation for provided models.
        :param models: callable, based on nn.Module (generators or discriminators)
        :param requires_grad: bool, disables or enables gradient computation upon calling .backward() method
        """
        for model in models:
            for param in model.parameters():
                param.requires_gard = requires_grad
