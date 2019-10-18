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

    @staticmethod
    def gender_head(in_features):
        # definition of head for gender model
        sequential_gender = nn.Sequential(nn.Dropout(p=0.3),
                                          nn.Linear(in_features, 256),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.3),
                                          nn.Linear(256, 128),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.3),
                                          nn.Linear(128, 1),
                                          nn.Sigmoid()
                                          )
        return sequential_gender

    @staticmethod
    def age_head(in_features):
        # definition of head for age model
        sequential_age = nn.Sequential(nn.Dropout(p=0.3),
                                       nn.Linear(in_features, 256),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(256, 128),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(128, 1),
                                       )
        return sequential_age
