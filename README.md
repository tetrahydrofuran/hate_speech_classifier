## Hate Speech Classifier  

This project was created as part of the Metis data science bootcamp.

Work in progress, readme will be updated upon completion.


### Repository Description
* The data .csv can be found [at this repository.](https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv)
  Place it within the data folder, or redirect the call to `pd.read_csv` in `main.py` to the appropriate location.
* Run `main.py` to generate non-deep learning models, according to the configuration settings defined at the top of the script.
* Run `keras-cnn.py` to train a convolutional neural network with 2 layers of convolution and pooling.

### Repository Structure
* `bin`
    * `processing`: Contains methods to process and normalize text, including part-of-speech classifier
    * `modeling`: Contains methods to generate non-deep learning models
    * `main.py`: Entry point for non-deep learning methods.  Configurable settings include:
        * `bool` to perform PCA or not before analysis, and `int` number of components
        * `bool` to convert 3-classes to binary classification problem
        * `bool` to force performing text normalization
        * `bool` to determine what type of model to generate
        * `str` description of run for metrics dataframe identification
        * `float` between 0 and 1 to define test size
        * `int` to serve as random seed for train-test splits for reproducibility
    * `keras-cnn.py`: Convolutional neural network to perform classification
* `data`: Dump of data, including processed dataframes, word dictionary to serve as reference, etc.
    * `models`: Generated serialized models
        * `v1-nonstratified`: Nonstratified sampling
        * `v2-stratified`: Stratified sampling
        * `v3-binary_stratified-class02`: Stratified sampling with class 1 and class 0 combined
        * `v4-pca_stratified-50components`: Stratified sampling with PCA to 50 components
        * `v5-pca_stratified-250components`: Stratified sampling with PCA to 250 components
    * `normalize-checkpoints`: Will be generated during text normalization steps, containing serialized dataframes before and after most time-consuming steps
* `docs`: Documentation associated with projects
* `img`: Generated images of model results


### Dependencies
* `sklearn`
* `pandas`
* `numpy`
* `nltk`
* `keras`

### If I Could Do It Over Again
* Save the `TfidfVectorizer()` objects fitted to my data
* Design with class imbalance in mind from the beginning
* Much better repository structure and organization