# VISOL_Classifier

## Purpose
- This is a classifier for the VISOL data competition.

- [link](https://dacon.io/competitions/official/236107/overview/description)

## Plan
1. Data Preprocessing
    - Preprocess Image to crop the half of image
    - Preprocess Image to resize the image to 224x224
    - Pickle the data
2. Modeling
    - Use pretrained ResNet model

[image]

## Augmentation : Random shuffle
    - Cut multiple images to create one image
    - There is not enough variation in the data, so we need to create more data.
[sample image]
