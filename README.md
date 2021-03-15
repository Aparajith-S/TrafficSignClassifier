# **Traffic Sign Recognition** 

author: Aparajith Sridharan  
date: 14:03:2021  

## Project: Build a Traffic Sign Recognition Program

Overview
---
In this project, knowledge of deep neural networks is used to make a convolutional neural network to classify German traffic signs. 
This involves supervised learning by training and validating a model so it can classify traffic sign images using the 
[German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  
After the model is trained, it was tried out on images of German traffic signs that was found on the internet.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dataset and Repository

1. Download the data set from here . The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.

[//]: # (Image References)

[image1]: ./visualizedataset.png "Visualization"
[image2]: ./histogram.jpg "Histogram"
[image3]: ./70kmph.jpg "GrayscaleAutocontrast"
[image4]: ./augment.png "augment"
[image5]: ./histafteraugmentation.png "histafter"
[image6]: ./results_40_Epochs.png "40Epochs"
[image7]: ./results_50_Epochs.png "50Epochs"
[image8]: ./deer.jpg "Traffic Sign 1"
[image9]: ./give_way.jpg "Traffic Sign 2"
[image10]: ./speed_limit.jpg "Traffic Sign 3"
[image11]: ./stop.jpg "Traffic Sign 4"
[image12]: ./roundabout.jpg "Traffic Sign 5"
[image13]: ./priority-road.jpg "Traffic Sign 6"
[image14]: ./turn_right.jpg "Traffic Sign 7"
[image15]: ./test.png "Traffic Sign Test"
[image16]: ./lenetviz.jpg "Traffic Sign Test"

---
### README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the data is distributed along the classes.

![alt text][image2]

![alt text][image1]

### Design and Test a Model Architecture

##### Observations  
- Quality of dataset    
As observed from the histogram, some classes have very few data.  
This will make the model perform poorly on test set containing these classes.  
Hence, augmentation needs to be done so that this histogram chart will look like one single rectangle i.e. all bins having almost same values.

- Quality of images  
The images are of varying brightness and of poor contrast.  
Some are hard to pick features from as they require some preprocessing like auto brightness or contrast corrections.  
One important note is that the images have colors. This could be an important feature.  
Surprisingly from the development it was found that the colors actually do not produce a notable effect on the end result.  

#### 1. Grayscaling, Auto contrasting using histogram equalization

On the side of data that is available but poor, the decision was made to use a grayscale conversion plus auto contrasting method using histogram equalization.
 
As a first step, I decided to convert the images to grayscale because as explained, the colors did not play a major role in classification performance and conversion means 3 channels to 1 channel. Thus, less computation load.  
Second step was to make a histogram equalization to correct poor contrast. This is done so that the CNN can pick up features better.
Here is an example of a traffic sign image before and after grayscaling plus histogram equalization.

![alt text][image3]

As a last step, I normalized the image data because the dataset is desirable to have values with a mean of 0 and a equal variance. 
it was done by finding the max and min of the array and doing (data-min)/(max-min).

I decided to generate additional data because as seen in the histogram before and the explanation that follows, the dataset has unequal distribution of data in the classes. It is advisable to have equal amount of data 
so that each class will have atleast a similar outcome in terms of accuracy.

To add more data to the the data set, I used the following techniques.
  - rotation of images: CNN is not rotation invariant hence a rotation of +/-20 degrees was considered : https://stackoverflow.com/questions/40952163/are-modern-cnn-convolutional-neural-network-as-detectnet-rotate-invariant/40953261#40953261
  - shearing/affine transformation on the images as there are images that can be taken at an angle that adds difficulty in recognition.
  - Translation was NOT considered because CNN is translation invariant. https://jvgemert.github.io/pub/kayhanCVPR20translationInvarianceCNN.pdf

Here is an example of an original image and an augmented image:
![alt text][image4]

The difference between the original data set and the augmented data set is the following :
- histogram of the data set now is uniform
![alt text][image5]

- a total of 2010 images in each class meaning the number of images are now 86430 for the training set.  
  Which is quite large and will help in tackling overfitting
 
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride, same padding, outputs 14x14x64	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling 2x2	    | 2x2 stride, same padding, outputs 5x5x16	    |
| Fully connected		| sizes - input: 400, output: 120               |
| RELU		            |              									|
| Dropout               |   30% dropout while training. 0% otherwise    |
| Fully connected		| sizes - input: 120, output: 84                |
| RELU		            |              									|
| Dropout               |   30% dropout while training. 0% otherwise    |
| Fully connected		| sizes - input: 84, output: 43                 |
| Softmax				|            									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters
Optimizer : ADAM  
Loss : categorical cross entropy  
Batch size : 128  
Epochs: 50  
Learning rate : 0.001  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Final model results:  
50 Epochs was chosen due to the acceptable margin of difference in train and validation accuracy and better test set accuracy.
The final performance of the model was found to be (considering the last 5 epochs)

| Epochs | training accuracy | validation accuracy | test set accuracy|  
|:------:|:-----------------:|:-------------------:|:----------------:|  
|   30   |         95        |         94.4        |       92.3       |
|   40   |         96.5      |         95.3        |       92.7       |
|   50   |         97.66     |         96.3        |       93.4       |

Anything above 50 epochs did not yield significant changes in the validation and testing accuracy. However, training accuracy was notably improved which indicated that more runs were leading to clear overfitting and hence was not pursued.

![alt text][image6]
The above plot shows the training and validation training and loss curves for 40 Epochs
 
![alt text][image7]
The above plot shows the training and validation training and loss curves for 50 Epochs

Both models satisfy the minimum validation score of 93%. infact, with the 50 epoch model and thanks to the augmented data used in training a test accuracy of 93.4% was achieved! 

If an iterative approach was chosen:  
* What was the first architecture that was tried and why was it chosen  
the basic LeNet-5 without dropout was tried as it was introduced in the course as well as in some papers where it was successfully used for traffic sign recognition.  

* What were some problems with the initial architecture  
The initial architecture overfits easily to the sparse data initially available  

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  
The architecture was not tweaked much except for Valid padding in conv layers and Same padding in max pool layers.  
dropout was added. this took care of one part of overfitting. the data augmentation did the rest.  
* Which parameters were tuned? How were they adjusted and why?  
batch size was increased/decreased and 128 was found to work well  
learning rate was set at 0.001  
epochs increased till 60, later settled to 50 after it was found where the model begins to stop learning features.   

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  
convolution layers are good at picking features at macro to micro level such as edges, orientation, patterns.  
dropout and batch normalizations are concepts that help against overfitting. introducing gaussian noise in the training input may as well help the cause.  
here dropout singlehandedly solved the problem so other methods were not required.

If a well known architecture was chosen:
* What architecture was chosen  
a modification of the LeNet 5  
* Why did you believe it would be relevant to the traffic sign application?  
https://link.springer.com/article/10.1007/s00500-019-04307-6  
one part was the lecture, other part was that, CV methods like HoGs were used in traffic sign detection before. This was efficiently replaced with CNNs as they could do the task better.  

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
The training set and the validation set will say how the model performed during the training. a small accuracy in both training and validation suggests that the model is underfitting. a large training accuracy and a low validation accuracy suggests that the model is overfitting. the loss curves can be examined to check where the overfitting happens and parameters can be adjusted based on that.  
hence, this model with the augmented data was able to produce a result with the training accuracy and validation accuracy being within +/-2% of each other's vicinity.   

The test set which is a blind set provides the final verdict on how the model might perform in the real world as the model is exposed to new data that was not the part of the training.  
This model was able to score quite well with a 93.4% accuracy.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12] ![alt text][image13] ![alt text][image14]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| animal crossing   	| animal crossing								|
| Yield					| Yield											|
| 30 km/h	      		| 30 km/h   					 				|
| Roundabout			| Roundabout          							|
| Priority Road			| Priority Road      							|
| Turn Right			| Turn Right        							|

The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%. 
This compares favorably to the accuracy on the test set of 93.4%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.
 ![alt text][image15]

- It was observed from the above probabilities that the system though performed perfectly, did have some uncertainities.  
- The signs that are triangle shaped are cause for confusion and arguably so, because there are a lot of traffic signs that closely resemble the same.  
- Moreover, a 32x32 image cannot capture all the minor features of the picture that distinguishes them apart.  
- It was observed that the wild animal crossing sign could be potentially misinterpreted as slippery roads.

- The Stop sign is interesting. it was taken at an angle, since the classifier was trained on skewed image it was able to do a bit better. same applies to the yield sign.  
But these are unique signs with unique shapes so, the classifier should be competent to pick up these features effortlessly.

The code for making predictions on my final model is located in the 52nd cell of the Ipython notebook.

 ### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

 ![alt text][image16]

The numerals in the traffic sign interested me, so I took the internal layers and visualized what it looks like in those layers. 
at a first glance, the network is in steps trying to read features. In the first layer, simple x-y derivatives are done, edges are activating the neurons. 
level 2 is more deeper patterns such as second derivatives, as i could roughly make minimas and maximas in the image.  
Of course it was also evident that the 32x32 image has proven to be quite insufficient to warrant further convolution layers as 
anything post the second layer is going to be just noisy output.

