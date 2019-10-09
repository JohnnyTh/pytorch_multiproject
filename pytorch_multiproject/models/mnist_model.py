import torch.nn as nn
from utils import same_padding_calc

class Nist_Net(nn.Module):

    def __init__(self):
        # super function to call init method of parent class nn.Module
        super(Nist_Net, self).__init__()

        """
        Role of the 'features' layer - recognition of various visual patterns in
        the images. The first CONV - ReLU - MaxPool layer detecs simplier features
        (vertical, horizontal edges, etc.), while the second such layer identifies
        more complex patterns(e.g. sahpes corresponding to different numbers)
        composed of the features.
        """
        conv_1_pad = same_padding_calc(28, 5, 1)
        conv_2_pad = same_padding_calc(14, 3, 1)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=conv_1_pad),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 3, stride=1, padding=conv_2_pad),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        """
        Role of the 'fc' layers - making a prediction which of the target numbers
        the visual pattern detected by the 'features' corresponds the most to.
        This can be done because FC layers have (as the name implies) connections
        with all the units from the previous layers.
        """
        self.fc_1 = nn.Sequential(nn.Linear(64 * 7 * 7, 256),
                                  nn.ReLU()
                                  )
        self.fc_2 = nn.Sequential(nn.Linear(256, 128),
                                  nn.ReLU(),
                                  )
        self.fc_3 = nn.Linear(128, 10)

        """Dropout is used to randomly zero some of the elements of layers' inputs
        in order to make the model generalize better and prevent co-adaptaion
        of neurons."""
        self.Dropout_1 = nn.Dropout(p=0.2)
        self.Dropout_2 = nn.Dropout(p=0.35)

    def forward(self, x):
        """Performs forward propagation

           Args:
                x (Tensor): a tensor containing input features.
        """
        x = self.features(x)
        x = x.view(x.shape[0], -1)

        # Perform dropout before and after the first fc layer
        x = self.Dropout_2(x)
        x = self.fc_1(x)
        x = self.Dropout_1(x)
        x = self.fc_2(x)
        x = self.Dropout_1(x)
        x = self.fc_3(x)
        return x