# Case Study
to be added

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


## Age Distrbution: 
![alt text][logo]

[logo]:https://github.com/sachin-econ/Case_Study/blob/main/download.png "Fig 1"




