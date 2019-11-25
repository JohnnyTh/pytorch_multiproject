# Cycle Generative Adversarial Networks
<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/69546988-6b239e80-0f9d-11ea-9ba0-f84489c5fe13.png" alt="torch" width="600"/>
</p>

Implementation of CycleGAN drawing architecture/training details from the [Repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Cycle GAN is a neural network architecture proposed by Jun-Yan Zhu et al., 2018 in the 
paper titled "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks". 

<p>As explained by the authors, CycleGAN consists of two generators (AB and BA) that perform translation of images from domain A to domain B and vice versa 
and two discriminators encourage the respective generators learn the correct translation.

<p>The paper introduced cycle consistency loss that imposes additional constraint on the mapping learned by generators 
that allows usage of unpaired training data.

<p>Tested on several datasets with varying results. 

###Horse - zebra dataset
One of the datasets, originally suggested by authors. The results in this implementation after 200 epochs of training were as follows:
<p align="center">
 <img src="https://user-images.githubusercontent.com/39649806/69551713-d4a7ab00-0fa5-11ea-9569-a1eeb1692625.jpg" alt="torch" width="600"/>
</p>

###MNIST - SVHN dataset
Description

###OLD - YOUNG dataset
Description