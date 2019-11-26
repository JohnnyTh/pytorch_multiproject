import torch.nn as nn
from utils import weights_inint_seq


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


class AgeGenderModelV2(nn.Module):

    def __init__(self, model):
        # super function to call init method of the parent class nn.Module
        super().__init__()

        # Freeze the model parameters
        for param in model.parameters():
            param.requires_grad = False
        # collect all resnet modules in a list
        modules = [module for module in model.children()]
        # remove the head of resnet
        modules = modules[:-1]
        # number of input features is determined number of convolutions in the last layer of resnet (512) multiplied by
        # expansion of used res blocks (for BasicBlock = 1, Bottleneck = 4)
        in_features = 512 * modules[4][0].expansion

        # get new heads for gender and age
        classifier_gender = self._gender_head(in_features)
        classifier_age = self._age_head(in_features)
        # initialize the weights of new layers using Xavier init
        weights_inint_seq(classifier_gender)
        weights_inint_seq(classifier_age)

        self.features = nn.Sequential(*modules)
        self.classifier_gender = classifier_gender
        self.classifier_age = classifier_age

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x_gender = self.classifier_gender(x)
        x_age = self.classifier_age(x)

        return x_gender, x_age

    @staticmethod
    def _gender_head(in_features):
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
    def _age_head(in_features):
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
