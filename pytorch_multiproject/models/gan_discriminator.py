import torch.nn as nn


class GanDiscriminator(nn.Module):

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
