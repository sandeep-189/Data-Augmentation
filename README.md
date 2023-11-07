# Data-Augmentation
Comparing a transormer GAN and a LSTM GAN for augmenting timeseries datasets.

The validation model notebooks train the classification models for verifying the GANs. The validation models have to be trained first before moving on to the training of GAN which can be done using the GAN training notebook.

The helper.py file contains all the one line loader functions for the training. The dataset.py file contains the function for cleaning the PAMAP2 and RWHAR datasets.

The helper.py file can be used to train custom validation models as well as custom generator/discriminator models by passing the appropriate model to the train functions as a parameter in the file.

# Prerequisites
To sun the code , install the python 3.8 environment using anaconda or miniconda. Using the conda environment, run "pip install -r requirements.txt" to install the needed python library versions.

The datasets are publicly available and will have to be downloaded from the official sites [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) [RWHAR](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) into a ./Data/{dataset_name} directory.

# Training
You can run the GAN_training.ipynb to train the model.
