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
### Version 0.4.0:
Implemented Word2Vec algorithm using Skip-Gram Negative Sampling (SGNS) model.
#### Skip-Gram Negative Sampling for word embeddings of specific corpus
<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/73250790-44adf980-41c0-11ea-928e-c1292c596249.jpg" alt="SGNS" width="600"/>
</p>

In this project a version of Skip-Gram Negative Sampling model for obtaining word embeddings was implemented as described in the original [paper](https://arxiv.org/abs/1310.4546).

<p align="justify">Training vectorized encodings of words to predict their most likely context in a sentence is a typical task of Word2Vec machine learning models. This is done 
using shallow neural networks with one hidden layer that are able to produce the desired word vectors while being fed one-hot or integer ID-encoded input words.
The main objective of such training is, interestingly, not to perform well on the task of predicting the words' context but to learn valuable 
word embeddings that can be later reused to encode words to train other common Natural Language Processing models such as LSTMs. 

<p align="justify">The value of word embeddings lies in the ability of encoded words to correlate with each other in multidimensional vector space.
Vector encodings of words with similar meaning will be positioned or oriented similarly thus allowing the model trained on these vectors 
to draw from them additional information and generalize better.

<p align="justify"> The architectures, most commonly used to train word embeddings, are Continuous Bag of Words (CBOW) and Skip-Gram models.
In this repo, Skip-Gram Negative Sampling was used to obtain the desired embeddings. The training objective was to differentiate between the 
true context of a word in a sentence and randomly generated negative context examples.

<p align="justify">To explore the ability of word vectors to reflect the relationships between the words from a certain corpus, 
the embeddings were trained on The Lord of the Rings trilogy by J. R. R. Tolkien. Some of the selected word vectors were 
converted into 2-D representations using t-SNE and the results can be seen below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/73247181-0103c180-41b9-11ea-80cd-3e098e883f00.jpg" alt="lotr_tsne" width="600" height="580"/>
</p>

<p align="justify"> In addition, cosine similarity was measured between the selected words and all the other words in vocabulary. 
The results (shown below) hinted that there is indeed a strong degree of correlation between the names of certain characters and objects.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/73250078-d452a880-41be-11ea-894c-78018181e7ae.jpg" alt="cos_sim" width="1200"/>
</p>

<p> If you wold like to check the visualization code, please refer to the utils/visualize_word2vec.ipynb file.

### Version 0.3.0:
Implemented [Mask R-CNN](https://arxiv.org/abs/1703.06870) for object detection using built-in torchvision model.
#### Object detection with Mask R-CNN
<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/72250821-2e6f3d80-3605-11ea-9983-18e04163d19b.jpg" alt="mask_r_cnn" width="600"/>
</p>

This project used the [Mask R-CNN model](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py)
provided in torchvision package and [Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/)
for bounding box regression and instance segmentation tasks.

<p align=”justify”>Since the model had already been implemented, the project was focused on creating utilities for model performance 
evaluation and visualisation of the results. The following components were implemented:

<ul>
    <li>Non-max suppression algorithm for output bounding boxes (utils/detection_evaluator.py)</li>
    <li>Mean Average Precision + Recall metrics calculation (utils/detection_evaluator.py)</li>
    <li>Mask and bounding box saver (utils/detection_evaluator.py)</li>
    <li>A number of transforms (random crop, gaussian blur) for data augmentation (data/custom_transforms.py)</li>
</ul>

The model was trained using hyperparameters from [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
A combination of custom transforms was used in order to improve performance of the model compared to the results in tutorial.
As shown in the image below, gaussian blur + horizontal flip and color jitter augmentation allowed for notable 
improvement over the default horizontal flipping.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/72254356-76de2980-360c-11ea-8ad4-9d3130a2fcf3.png" alt="mask_val" width="500"/>
</p>

The outputs of the trained model can be seen below (green - ground truth bounding boxes, 
red - predicted bounding boxes; masks are displayed as coloured areas overlaid on top of persons' silhouettes):

<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/72255424-5c597f80-360f-11ea-9fcc-0fa31f7e1795.png" alt="mask_val" width="1000"/>
</p>

### Version 0.2.0:
Implemented CycleGAN for several tasks based on the following [Paper](https://arxiv.org/pdf/1703.10593.pdf). 
The tasks include unpaired image-to-image translation for horse-zebra, mnist-svhn, and old-young datasets. 
#### Cycle Generative Adversarial Networks
<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/69902337-fb465700-1394-11ea-9a9b-de1b582c86ec.png" alt="cylegan" width="600"/>
</p>

Implementation of CycleGAN drawing architecture/training details from the [Repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Cycle GAN is a neural network architecture proposed by Jun-Yan Zhu et al., 2018 in the 
paper titled "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks". 

<p>As explained by the authors, CycleGAN consists of two generators (AB and BA) that perform translation of images from domain A to domain B and vice versa 
and two discriminators that encourage the respective generators to learn the correct translation.

<p>The paper introduced cycle consistency loss that imposes additional constraint on the mapping learned by generators 
that allows usage of unpaired training data.

<p>Tested on several datasets with varying performance. 

#### Horse - Zebra dataset
One of the datasets, originally suggested by authors. The results in this implementation after 200 epochs of training were as follows:
<p align="center">
 <img src="https://user-images.githubusercontent.com/39649806/69551713-d4a7ab00-0fa5-11ea-9569-a1eeb1692625.jpg" alt="torch" width="600"/>
</p>

##### MNIST - SVHN dataset
This task was done using built-in torchvision MNIST (Modified National Institute of Standards and Technology database) 
and SVHN (Street View House Numbers) datasets.
<p align="center">
 <img src="https://user-images.githubusercontent.com/39649806/69553468-eb9bcc80-0fa8-11ea-9869-cc637ce26293.jpg" alt="torch" width="400"/>
</p>


##### Old - Young dataset
Image-to-Image translation was done on two subsets of human faces (old and young) extracted from 
[IMDB - WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). The datasets were preprocessed and 
balanced by gender prior to training in order to improve the quality of the final results (see the details of preprocessing at
pytorch_multiproject/utils/age_gender_preprocessing.py).
<p align="center">
 <img src="https://user-images.githubusercontent.com/39649806/69608464-2eed4e00-1030-11ea-9ef9-fe1a339627d0.jpg" alt="torch" width="600"/>
</p>

### Version 0.1.0:
Implemented all basic features - basic and generic classes for trainers, unit tests for important modules, 
logging of results. Generic trainer was implemented with serialization, deserialization and cycling through epochs methods. 
Additionally, two custom projects were added. First project is a simple NN for MNIST classification task 
(based on the Author's [previous work](https://github.com/JohnnyTh/MNIST_convnet_pytorch)). The second project is described below.
#### Transfer learning for age and gender classification
<p align="center">
 <img src="https://user-images.githubusercontent.com/39649806/73367790-afdaf700-42b8-11ea-867a-5bbc5ddf7070.jpg" alt="age_gen" width="800"/>
</p>

The project uses transfer learning to re-purpose a pre-trained model for age regression and gender classification task. Source images 
for training were taken from IMDB-WIKI dataset (more precisely, only WIKI part). [Link to the dataset description](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
[Link to the dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar). 
As can be seen from the image above, two heads are trained together and the layer's weights are updated based on the total of the system defined as 
the sum of age regression task loss and gender classification task loss.


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
|   └── word2vec_dataset.py - dataset class for word2vec model
│
├── logger/ - module for logging
│   ├── logger.py
│   └── logger_config.json
|
├── models/ - models, losses, and metrics
│   ├── age_gender_model.py - Neural network model for age and gender classification
|   ├── cycle_GAN.py - NN model, optimizer, lr_scheduler for Cycle Generative Adversarial Network
|   ├── mask_r_cnn_model.py - Mask R-CNN from torchvision package
│   └── mnist_model.py - Neural network model for MNIST dataset classififcation
│   └──word2vec.py - Skip-Gram Negative Sampling model
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
|   └── mask_r_cnn/ - Mask R-CNN for object detection
│   |   ├── test.py
│   |   ├── train.json
│   |   └── train.py
│   |
|   └── mnist/ - MNIST dataset classififcation model
│   |    ├── train.json
│   |    └── train.py - main script to start training
│   |
|   └── word2vec/ - Word embeddings using SGNS
|        ├── train.json
|        └── train.py
|
├── resources/ - default directory for storing input datasets
|
├── saved/ - trained models and test data are saved here by default
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
    └── visualize_word2vec.ipynb - visualization for SGNS model.
    ├── age_gender_preprocessing.py - downloads and pre-processes the data for gender age classifier and cycle GAN
    ├── gan_horses dataset.py - downloads horse - zebra dataset for cycle GAN
    └── detection_evaluator.py - metrics and mask + bbox saver for Mask R-CNN
    └── word2vec_dataset.py - text preprocessing for LOTR dataset
    └── word2vec_dataset_movies.py - text preprocessing for movie lines dataset
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