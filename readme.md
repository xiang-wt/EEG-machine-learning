# Interpretable Machine Learning Model for Predicting TD and Severity Based on EEG



## Usage



### Step 1. Preprocessing (MATLAB)

Run the MATLAB scripts to preprocess EEG and extract features:  

```matlab

run('eeg_preprocessing.m')

run('feature_extraction.m')

```



### Step 2. Modeling (Python)

Train and evaluate the model:  

```bash

python dataload.py 

python train.py

```



