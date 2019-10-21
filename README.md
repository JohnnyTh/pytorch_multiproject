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
* More models and custom implementations of train.py;
* Improve compatibility with different OS;
* Add more optional arguments for train.py and fix existing ones (especially checkpoint loading).

## Structure:
```
pytorch-multiproject/
|
├── data/ - data-related classes and operations
│   ├── custom_transforms.py - custom tranforms to be used with Dataset class
|   ├── generic_dataset.py - implements generic class for basic data operations (inherits from PyTorch Datset)
|   └── gender_age_dataset.py - implements custom class for age and gender classfication task (inherits from generic)
│
├── logger/ - module for logging
│   ├── logger.py
│   └── logger_config.json
|
├── models/ - models, losses, and metrics
│   ├── age_gender_model.py - Neural network model for age and gender classification
│   └── mnist_model.py - Neural network model for MNIST dataset classififcation
│
├── projects/ - actual DL projects created using the tools provided in repo
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
├── saved/ - trained models are saved here
│
├── tests/ - unit tests
|
├── trainers/ - base, generic, and custom trainers
|   ├── age_gender_trainer.py
|   ├── base_trainer.py 
|   ├── generic_trainer.py
|   └── mnist_trainer.py
|
│
└── utils/ - small utility functions
    ├── util.py
    └── ...
```
#### Running unit tests
Example:
```
pytest base_trainer.py
```
### Dependencies
```
python >= 3.7.0
torch >= 1.2.0
torchvision
numpy >= 1.17
pandas >= 0.24.2
pytest >= 4.3.1
scikit-image >= 0.14.2
scikit-learn >=  0.20.3
mock >= 3.0.5
```
### Author

Maksym Tatariants [@JohnnyTh](https://github.com/JohnnyTh ) 