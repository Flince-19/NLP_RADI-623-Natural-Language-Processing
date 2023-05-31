# Lung nodule classification using CNN
Repository for lung nodule classifcation using CNN as part of RADI 605: Modern Machine Learning assignment. See the notebook in 'src\lung_nodule_CNN.ipynb' for the code. The code was written in Pytorch. 

## Notes
- Put the data in the data folder. The data structure should be:

```
data/
|
├── train/
|   ├── image1.jpg
|   ├── image2.jpg
|   ├── ...
|
├── val/
|   ├── image1.jpg
|   ├── image2.jpg
|   ├── ...
|
├── test/
|   ├── image1.jpg
|   ├── image2.jpg
|   ├── ...
|
├── testlabels.txt
├── trainlabels.txt
└── vallabels.txt
```

- The notebook utilize various modules for the pipeline. The codes, including the training loop, were modified from the 
[Zero to Mastery Learn PyTorch for Deep Learning course](www.learnpytorch.io).
- Torch version 2.0 and torchvision version 0.15 were used. However, torch version >= 1.12.0 should be enough.
- The cells for training is included in the notebook. However, the training has already been completed and the models were saved in trained_model folder, which will be loaded for evalaution of the test set in the next cell. The module used for training automatically plot the loss curve and automatically save the model with the least loss. Do not run the cell if you do not want to retrain again.
- Seeds were set at various steps with the set_custom_seed module.

# Step Explanations
## Step 1-3
The dataset was loaded into the dataloader by the data_setup_module. Detailed exaplantion of the module can be found in the comment corresponding .py file. Briefly, the relative path for the images were retreived and compared to the 'image' column in the txt files. Only image with matching path were kept (as there are missing labels in the training data). The filtered path was used to retrieve each image and the correspond labels were retrieved from the 'label' column with the mathcing row. The data_setup module for creation of dataloader was written so as to output the path to the image from the dataset as well.

## Step 4
Before creating each dataset, transform must be specified. In this case, the image is resized to the desired size with transforms.Resize() and convert to tensor in the range of  [0.0, 1.0] with transforms.ToTensor() to be fed to the model. For the training set, data augmentation was done with RandomHorizontalFlip(). 

As the image is a cropped square image of a CT slice containing lung nodules, it does not actually contains important side-sensitive landmark structure. As such, vertical/horizontal flip and rotation should be applicable. Cropping is a bit risky as (as far as I know because I have not exhaustively checked) there is no garauntee that location of the nodule is in the same place, so random crop might accidentally crop out the nodule because some malignant nodules can be very tiny in some image. Anything involving zooming should not also be done as size is a predictor of malignancy. Blurring or shapening should alno not be used since spiculated nodules have higher chance of malignancy and blurring/shapening might change that features. Brightness might also be applicable but not contrast.

In the notebook, I only implemented horizontal flip because it was the first obvious choice that came to mind and after experiment, the performance on the test set was quite good already and I did not implement other augmentation. If I were to did another experiment, I would try vertical flip, and rotation. Another method to consider would be TrivalAugment which is a parameter-free method which applies simple transform such as flip, rotation, color jitter, resize and crop with random parameter without the need for parameter optimization. It does perform some operation which violates my hypothesis (copping, resize) but nevertheless is an interesting technique to try.

## Step 5-9
The first model architecture was a SimpleCNN_V1. It employed Conv2d layer followed a ReLu activation and Maxpooling to reduce the dimension. The output size calculation for Conv2D was demonstrated in the notebook. The first Conv2D block produce 16 feature maps which were sent to another block and produce another 32 feature mapes before being flattened into a dense layer. The loss function was BCEwithlogitloss as the problem is binary classification and BCEwithlogitloss is reportedly more stable than regular BCE. After the logits are computed, it was passed to sigmoid and rounded to obtained the label.The model converged pretty well and the performance was pretty good for a starting model with 8,673 parameters.

As the class is imbalanced, the metric used for evaluation in this experiment were balanced accuracy and F1 score. However, F1 does not give importance to classifying negative example, it is used where classifying true positive is more important. In this case, positive class is malignancy and false negative can means death while false positive might just means redundant biopsy, thus I prefer to give use F1 in this case.  Another suitable metric is the precision-recall  curve which takes into account data imbalanced as well while providing visualization of tradeoff between precision and recall with differnt threshold and can be used to select the best threshold. Performance of predicting all sample as negative was also provided as a baseline for comparison.

To further improve the model, V2 and V3 were created.
- V2's design was based on assumption that there are more complex features to be extracted and such, 2 more Conv2D layers were stacked which output 64 feature maps each. Indeex, the performance improved but the parameter exploded to 60,481 parameters.
- V3's design introduce batch normalizaton and dropout layer.  Batch normalizaton normaliza the output of the prvious layer, reducing covariate shift and should stabilize the training and improve performance. Dropout layer randomly deactivate neurons with certain probability to prevent overfitting and improve generalization. After introduction of those two elements, performance indeed improve again with only marginal increase parameter (60,481 -> 60,833).
- As extra, I tried implementing [TinyVGG](https://poloclub.github.io/cnn-explainer/) used in CNN EXPLAINER and CS231n: Deep Learning for Computer Vision. The architecture was actually almost the same as SimpleCNN_V1 but with a total of 4 Conv2D layers. As I defined the hidden unit to be only 10, it only produce 10 feature maps for each Conv2D layer and has only 4,271 parameters, half of SimpleCNN_V1. It actually performed equally, or better, than SimpleCNN_V1, suggesting that deeper network with fewer parameters is better than wider but shallower network.

Finally, for transfer learning, I used [EfficientNet_B0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html). The pre-trained weight was from the original [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) which was pre-trained on ImageNet-1K dataset. All layers of the model was unfrozen and trained. EfficientNet_B0 demonstrated the best performance with the least epoch (at epoch 3 of training). It employs Mobile Inverted Bottleneck Convolution (MBconv) block whcih allows [fewer parameter with comparable performance](https://towardsdatascience.com/residual-bottleneck-inverted-residual-linear-bottleneck-mbconv-explained-89d7b7e7c6bc) and SiLu function which can output negative values as opposed to ReLu and has shown to acheive [better performance](https://www.sciencedirect.com/science/article/pii/S0893608017302976) than ReLu. EfficientNet_B0 also has the largest amount of parameter (4,008,829) among all the models used in the experiment. 


As can be seen from the experiment result, factors which leads to better performance might includes 
- More parameters. There was diminishing return with increasing parameter but the more parameters, the better the performance was.
- Deeper network (as seen with improvement from SimpleCNN_V1 to SimpleCNN_V2) which might allows for extraction of more complex features. Deeper network also seems to perform better than shallow but wide model (as seen with difference in SimpleCNN_V1 and TinyVgg_V1).
- Regularization to prevent overfitting and stabilization of network (as seen with improvement from SimpleCNN_V2 to SimpleCNN_V3). It improves performance with relatively little increase in parameters.
- Transfer learning. While EfficientNet_B0 utilized more complex Conv block and activation function and having larger parameters, achieving the least loss with just 3 epochs of training. The knowledge learned from ImageNet-1K seems to be transferable to classification of CT scan of lungs nodule as well. As the ImageNet-1K was on a relatively different data, a model pre-trained on similar data(such as CT scan from other organ) might be able to produce better results.






