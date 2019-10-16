import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.append(ROOT_DIR)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from projects.mnist.mnist_model import MnistNet
from trainers.mnist_trainer import MnistTrainer

resources_dir = os.path.join(ROOT_DIR, 'resources', 'mnist')

mnist_trainset = datasets.MNIST(root=resources_dir, train=True, download=True, transform=transforms.ToTensor()
                                )
mnist_testset = datasets.MNIST(root=resources_dir, train=False, download=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=128, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=128, shuffle=False, num_workers=0)

dataloaders = {'train': trainloader, 'val': testloader}

# Define metrics
metrics = {'epoch': 0, 'loss': 10., 'acc': 0.0 }
# Define number of epochs
epochs = 40
# Create an instance of model
model = MnistNet()
# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


def f(epoch): return 0.89 ** epoch  # Set a learning rate scheduler
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


session = MnistTrainer(dataloaders, scheduler, ROOT_DIR, model, criterion, optimizer, metrics, epochs)

if __name__ == '__main__':
    session.train()
