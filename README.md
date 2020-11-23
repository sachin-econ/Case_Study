# Case Study
to be added

## Data
IMDB-WIKI is the largest publicly available dataset of face images with gender and age labels for training and testing. There are 460,723 face images from 20,284 celebrities from IMDb and 62,328 from Wikipedia, being thus 523,051 in total. Rothe et al. crawled the dataset for age prediction and tackled the estimation of apparent age with CNN using the VGG-16 architecture, which they made public in 2016.\n\
The complete data sets from the IMDB-WIKI project is large (272 GB). We use much smaller subsets of face-only data for our study. The image metadata stored in the MATLAB binary database was converted using SciPy and added to Pandas DataFrame after applying the below conditions.
face_score field - higher the better - started with a minimum value of 0.0 and skipped over records with a face_score value of Inf.(no face)
also skip records with second_face_score value(not equal to NaN), or where the image was missing.
Please refer to 'function.py' in the GitHub link for age, gender, and age bucket calculation/definitions.
We remove duplicates and those records with ages below 0 or over 100. Key statics below:



## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
