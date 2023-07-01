# VISOL_Classifier

## Purpose
- This is a classifier for the [VISOL data competition](https://dacon.io/competitions/official/236107/overview/description).
## Plan
1. Data Preprocessing
    - Preprocess Image to crop the half of image
    - Preprocess Image to resize the image to 224x224
    - Pickle the data
2. Modeling
    - Use pretrained ResNet model


## Augmentation : Random shuffle
    - Cut multiple images to create one image
    - There is not enough variation in the data, so we need to create more data.
<p align="middle">
  <img src="https://github.com/MinTagg/VISOL_Classifier/assets/98318559/b5d92e51-c75d-4a77-8f35-470a51fe1e03">
  <img src="https://github.com/MinTagg/VISOL_Classifier/assets/98318559/73b91fd4-ecb2-4a11-a882-1e01a4a01dbf">
</p>

## Augmentation : Others
  - HSV
  - Rotation
  - Flip
  - Gaussian Noise
  - translation
  - shear

## Result
<p align="middle">
  <img src="https://github.com/MinTagg/VISOL_Classifier/assets/98318559/9a7de48f-45ca-483c-93c7-dce065d707d6">
</p>

- Detect unlabeled image into 'unknown' class
