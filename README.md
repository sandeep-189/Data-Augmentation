# Data-Augmentation
Comparing a transormer GAN and a LSTM GAN for augmenting timeseries datasets.

The validation model notebooks train the classification models for verifying the GANs. The validation models have to be trained first before moving on to the training of GAN which can be done using the GAN training notebook.

The helper.py file contains all the one line loader functions for the training. The dataset.py file contains the function for cleaning the PAMAP2 and RWHAR datasets.

The helper.py file can be used to train custom validation models as well as custom generator/discriminator models by passing the appropriate model to the train functions as a parameter in the file.

# Requirements
python >= 3.8 

pytorch >= 1.7 

pytorch-lightning >= 1.2.1 

tensorboard >= 2.2 
