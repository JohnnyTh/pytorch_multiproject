## Object detection with Mask R-CNN
<p align="center">
  <img src="https://user-images.githubusercontent.com/39649806/72250821-2e6f3d80-3605-11ea-9983-18e04163d19b.jpg" alt="mask_r_cnn" width="600"/>
</p>

This project used the [Mask R-CNN model](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py)
provided in torchvision package and [Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/)
for bounding box regression and instance segmentation tasks.

<p>Since the model had already been implemented, the project was focused on creating utilities for model performance 
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
