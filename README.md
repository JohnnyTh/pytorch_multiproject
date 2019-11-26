# Pytorch Multproject # 
<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/67198092-e6c28680-f406-11e9-8bb2-9b262787b82d.png" alt="torch" width="150"/>
</p>

The main idea behind this project is to create a generic foundation upon 
which customized pytorch projects can be built. This means implementing abstract 
base classes as well as generic classes for the key elements found in any PyTorch projects 
such as Dataset and Trainer classes. The repo includes several implemented Deep Learning 
projects demonstrating usage of these concepts in practice.

## Version history:
##### Version 0.2.0:
Implemented CycleGAN for several tasks based on the following [Paper](https://arxiv.org/pdf/1703.10593.pdf). 
The tasks include unpaired image-to-image translation for horse-zebra, mnist-svhn, and old-young datasets. For more info
look at the readme file at pytorch_multiproject/CycleGAN in this repo.
##### Version 0.1.0:
Implemented all basic features - basic and generic classes for trainers, unit tests for important modules, 
logging of results. Generic trainer was implemented with serialization, deserialization and cycling through epochs methods. 
Additionally, two custom projects were added. First project is a simple NN for MNIST classification task 
(based on the Author's [previous work](https://github.com/JohnnyTh/MNIST_convnet_pytorch)). The second project 
uses transfer learning to re-purpose a pre-trained model for age and gender classification task. Source images 
for training were taken from IMDB-WIKI dataset (more precisely, only WIKI part).
[Link to the dataset description](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
[Link to the dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar)
##### TODOs:
* More models (semantic segmentation, object detection, image denoising, NLP models).

### Structure: 
```
pytorch-multiproject/
|
├── data/ - data-related classes and operations
│   ├── custom_transforms.py - tranforms for use with Dataset class
|   ├── cycle_gan_age_dataset.py - class for image-to-image translation using GAN for old-young dataset
|   ├── cycle_gan_dataset.py - class for image-to-image translation from domain A to B and 
|   |                          vice versa using GAN (inherits from generic)
|   ├── cycle_gan_dataset_small.py - class for image-to-image translation using GAN for MNIST-SVHN datasets
|   |                                (inherits from generic)
|   ├── gender_age_dataset.py - class for age and gender classfication task (inherits from generic)
|   └── generic_dataset.py - generic class for basic data operations (inherits from PyTorch Datset)
│
├── logger/ - module for logging
│   ├── logger.py
│   └── logger_config.json
|
├── models/ - models, losses, and metrics
│   ├── age_gender_model.py - Neural network model for age and gender classification
|   ├── cycle_GAN.py - NN model, optimizer, lr_scheduler for Cycle Generative Adversarial Network
│   └── mnist_model.py - Neural network model for MNIST dataset classififcation
│
├── projects/ - actual DL projects created using the tools provided in repo
|   ├──CycleGAN/ - implemetations of Cycle Generative Adversarial Network
|   |   ├── age.json - hyperparameters for old - young task
|   |   ├── age_test.py - test script for old - young task
|   |   ├── age_train.py - train script for old - young task
|   |   ├── small.json - hyperparameters for MNIST - SVHN task
|   |   ├── small_test.py - test script for MNIST - SVHN task
|   |   ├── small_train.py - train script for MNIST - SVHN task
|   |   ├── test.py - test script for general for horse - zebra task
|   |   ├── train.json - hyperparameters for horse - zebra task
|   |   └── train.py - train script for horse - zebra task
|   |
│   ├── gender_age_model/ - model for age and gender classification task
│   |   ├── train.json
│   |   └── train.py - main script to start training
|   |
|   └── mnist/ - MNIST dataset classififcation model
|        ├── train.json
|        └── train.py - main script to start training
|
├── resources/ - default directory for storing input datasets
|
├── saved/ - trained models and test data are saved here
│
├── tests/ - unit tests
|
├── trainers/ - base, generic, and custom trainers
|   ├── age_gender_trainer.py
|   ├── base_trainer.py
|   ├── cycle_gan_trainer.py
|   ├── generic_trainer.py
|   └── mnist_trainer.py
|
│
└── utils/ - small utility functions
    ├── util.py
    ├── age_gender_preprocessing.py - downloads and pre-processes the data for gender age classifier and cycle GAN
    ├── gan_horses dataset.py - downloads horse - zebra dataset for cycle GAN
    └── ...
```

#### Running unit tests
Example:
```
pytest base_trainer.py
```
### Prerequisites
* Python 3
* CPU or NVIDIA GPU supporting CUDA  and CuDNN
### Dependencies
```
torch >= 1.2.0
torchvision
numpy >= 1.17
pandas >= 0.24.2
pytest >= 4.3.1
requests >= 2.22.0
scikit-image >= 0.14.2
scikit-learn >=  0.20.3
mock >= 3.0.5
tqdm >= 4.37.0
```
### Author

Maksym Tatariants [@JohnnyTh](https://github.com/JohnnyTh) 