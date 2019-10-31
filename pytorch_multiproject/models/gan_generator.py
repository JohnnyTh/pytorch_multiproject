import torch.nn as nn


class ResBlock(nn.Module):

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

        if not self.skip_relu:
            # missing relu added!
            out = self.relu(out)
        return out


class GanGenerator(nn.Module):

    def __init__(self, skip_relu=False):
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

        resblocks = [ResBlock(skip_relu=skip_relu)] * 9

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

    def forward(self, input):
        return self.model(input)