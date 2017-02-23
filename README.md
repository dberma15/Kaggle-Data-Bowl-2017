# Kaggle-Data-Bowl-2017

This is my approach to the Kaggle Data Bowl 2017. 

Given that the number of slices for each patient varies, I figured some type of recurrent neural network (RNN) would be a good approach. In order to do this, I needed a feature extractor. Training a convolutional neural network (CNN) would take far more data than I have from the Data Bowl, so I decided to start with the CIFAR-10 data using two classes (6 and 9) from the data set in order to try to build a quick CNN. This required that I increase the size of the images to 150 by 150 pixels and decrease the size of the medical images from 512 by 512. This worked reasonably well  (error rate of 0.60035). 

My next approach is to use the VGG_16 CNN. I remove the last two drop out and dense layers and use that as a feature extractor. I wanted to remove the third to last dense layer but I didn't have the RAM to process those files as the feature vector would be 25088 elements. 
