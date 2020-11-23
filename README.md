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

[Age]:https://github.com/sachin-econ/Case_Study/blob/main/download.png "Fig 1 Age Distrbution"  
  
**Size of the bucket “15 to 25 years old”:** 39369   
**Percentage of this population is "30 year old males":** 1.87% 

## Age Estimation Algorithm
Age estimation has become more relevant in recent years with applications across fields for intelligent ad delivery, bioassays science, biostatistics, electronic customer relationship management, recognition, law enforcement, security controls, demographic census, human-computer interaction, and other fields.  
A traditional algorithm typically performs a set of mathematical operations on the raw pixel values of the image called feature extraction. These features are combined to make a final decision about the image using a combination of manually tuned parameters or thresholds. In comparison, deep learning offers dramatic performance improvements by learning from examples specific to the classification problem at hand, and with both feature extraction and decision are learned systems do not require endless tuning to adapt to changing conditions.   
In practice, the use of the multi-scale deep convolutional neural network has grown due to its impressive performance in end-to-end age regressor. For the IMDB- WIKI dataset, the first and second-edition winners (Rothe et al and Antipov et al ) and the ChaLearn LAP competition on Apparent Age Estimation use VGG-16 as base architecture.  
In our method, we chose the Dense Convolutional Network (specifically DenseNet 121), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - DenseNet network has L(L+1)/2 direct connections.   

![alt text][DenseNet 121]

[DenseNet 121]:https://github.com/sachin-econ/Case_Study/blob/main/dense%20121.png "Fig 2 DenseNet 121"  

An output of the previous layer acts as an input of the second layer by using composite function operation. This composite operation consists of the convolution layer, pooling layer, batch normalization, and non-linear activation layer.   
DenseNets have several compelling advantages: they alleviate the vanishing gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. DenseNets obtain significant improvements over the state-of-the-art on most standard datasets, whilst requiring less computation to achieve high performance.  
Additionally, we approach age estimation as a classification (bucketing ages) problem rather than regression, which helps us navigate data issues and computational limitations.
This classification approach helps us to control the effects of the skewness of the dataset towards the middle ages, and the distinction between apparent and real age addressed in many recent papers. Moreover, in real-world scenarios, a classification strategy is better for targeting both businesses and policymakers.  



