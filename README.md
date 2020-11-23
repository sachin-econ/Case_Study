# Case Study
Age Estimation using IMDB-WIKI.

## Data
IMDB-WIKI is the largest publicly available dataset of face images with gender and age labels for training and testing. There are 460,723 face images from 20,284 celebrities from IMDb and 62,328 from Wikipedia, being thus 523,051 in total. Rothe et al. crawled the dataset for age prediction and tackled the estimation of apparent age with CNN using the VGG-16 architecture, which they made public in 2016.  
The complete data sets from the IMDB-WIKI project is large (272 GB). We use much smaller subsets of face-only data for our study. The image metadata stored in the MATLAB binary database was converted using SciPy and added to Pandas DataFrame after applying the below conditions.  
**face_score field** - Higher the better - started with a minimum value of 0.0 and skipped over records with a __face_score__ value of Inf.(no face) or where the image was missing.  
__second_face__ - Also skip records with second_face_score value(not equal to NaN).  
Please refer to [function.py](https://github.com/sachin-econ/Case_Study/blob/main/functions.py) for age, gender, and age bucket calculation/definitions.  
We also remove duplicates and those records with ages below 0 or over 100. Key statics below:  

| Tables | Gender|  Age  | Age Group|
| ------ |------:|------:|---------:|
| count  |209946 |209946 |    209946|
| mean   |0.58   |36.69  |      3.71|
| std	   |0.49	 |13.73	 |      2.16|
|min	   |0	     |0	     |         0|
|25%     |0	     |27     |       	2|
|50%     |1	     |35	   |         4|
|75%	   |1	     |44     |         5|
|max	   |1	     |100	   |         9|


## Age Distrbution 
![alt text][Age]

[Age]:https://github.com/sachin-econ/Case_Study/blob/main/resources/download.png "Fig 1 Age Distrbution"  
  
**Size of the bucket “15 to 25 years old”:** 39369   
**Percentage of this population is "30 year old males":** 1.87% 

## Age Estimation Algorithm
Age estimation has become more relevant in recent years with applications across fields for intelligent ad delivery, bioassays science, biostatistics, electronic customer relationship management, recognition, law enforcement, security controls, demographic census, human-computer interaction, and other fields.  
A traditional algorithm typically performs a set of mathematical operations on the raw pixel values of the image called feature extraction. These features are combined to make a final decision about the image using a combination of manually tuned parameters or thresholds. In comparison, deep learning offers dramatic performance improvements by learning from examples specific to the classification problem at hand, and with both feature extraction and decision are learned systems do not require endless tuning to adapt to changing conditions.   
In practice, the use of the multi-scale deep convolutional neural network has grown due to its impressive performance in end-to-end age regressor. For the IMDB- WIKI dataset, the first and second-edition winners (Rothe et al and Antipov et al ) and the ChaLearn LAP competition on Apparent Age Estimation use VGG-16 as base architecture.  
In our method, we chose the Dense Convolutional Network (specifically DenseNet 121), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - DenseNet network has L(L+1)/2 direct connections.   

![alt text][DenseNet 121]

[DenseNet 121]:https://github.com/sachin-econ/Case_Study/blob/main/resources/dense%20121.png "Fig 2 DenseNet 121"  

An output of the previous layer acts as an input of the second layer by using composite function operation. This composite operation consists of the convolution layer, pooling layer, batch normalization, and non-linear activation layer.   
DenseNets have several compelling advantages: they alleviate the vanishing gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. DenseNets obtain significant improvements over the state-of-the-art on most standard datasets, whilst requiring less computation to achieve high performance.  
Additionally, we approach age estimation as a classification (bucketing ages) problem rather than regression, which helps us navigate data issues and computational limitations.
This classification approach helps us to control the effects of the skewness of the dataset towards the middle ages, and the distinction between apparent and real age addressed in many recent papers. Moreover, in real-world scenarios, a classification strategy is better for targeting both businesses and policymakers.  

#### Algorithm Tradeoffs
Even with selective classification, data imbalance at the lower and higher end of the data still affects the quality of prediction. In other models trained using the same dataset, researchers have used approaches like a) equalizing the age distribution; i.e., randomly ignoring some of the images of the most frequent ages and b)manually collected a private dataset of 5723 children images in the 0-12 age category and training them separately.   
Additionally, use of the compiler cropped dataset face and quality parameter lead to loss of around 40K images, used in other architectures trained using raw data from the same source.  
DenseNet uses a lot more memory when compared to earlier architecture (like ResNet), as the tensors from different layers are concatenated together. Even though DenseNets are advanced models they still suffer some of the limitations of deep learning. 

#### Loss functions
The focal loss was implemented in the Focal Loss for Dense Object Detection paper by He et al. Focal loss is a dynamically scaled cross-entropy loss, where the scaling factor automatically decays to 0 as the confidence in the correct class increases.  
Models trained using Binary Cross-Entropy loss requires the model to be confident about what is predicting. Whereas, what Focal Loss does is that it makes it easier for the model by giving the model a bit more freedom to take some risk when making predictions.   
Particularly important when dealing with highly imbalanced datasets because in some cases, we need to model to take a risk and predict something even if the prediction turns out to be a False Positive (like minors in a bar). Therefore, Focal Loss is particularly useful in cases where there is a class imbalance like IMDB Wiki Dataset.  

The Focal Loss is mathematically defined as:  
![alt text][Focal Loss]

[Focal Loss]:https://github.com/sachin-econ/Case_Study/blob/main/resources/Focal%20Loss%20Fn.PNG "Fig 3 Focal Loss Equation"  

There are two adjustable parameters for focal loss.  
*  The focusing parameter γ(gamma) smoothly adjusts the rate at which easy examples are down-weighted. When γ = 0, focal loss is equivalent to categorical cross-entropy, and as γ is increased the effect of the modulating factor is likewise increased (γ = 2 works best in experiments).
*  α(alpha): balances focal loss, yields slightly improved accuracy over the non-α-balanced form.

Implementation of Focal Loss code in [utility.py](https://github.com/sachin-econ/Case_Study/blob/main/utility.py).

#### Model Dependency
No, Deep learning systems are “trained” to perform identification tasks by being presented with many examples of pictures, objects, or scenarios that humans have already labeled “correct” or “incorrect". These labeled examples or training data play a key role in determining the overall accuracy of these systems.  
The training data in our model is unevenly distributed, with fewer records in the younger and older age groups. Thus our model is more likely to misclassify a person from these groups. Similarly, the dataset also lacks diversity among gender and ethnic groups and may lack the ability to predict age groups accurately in these less represented demographics.

 
