import torch.nn as nn


class AgeGenderModel(nn.Module):

    def __init__(self):
        # super function to call init method of the parent class nn.Module
        super(AgeGenderModel, self).__init__()
        self.features = None
        self.classifier_gender = None
        self.classifier_age = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x_gender = self.classifier_gender(x)
        x_age = self.classifier_age(x)

        return x_gender, x_age
