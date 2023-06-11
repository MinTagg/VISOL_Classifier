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

<p align="center">
  <img src="https://github.com/MinTagg/VISOL_Classifier/assets/98318559/7df47c76-3850-4b60-bb40-c2e059c46a6a">
</p>

## Augmentation : Random shuffle
    - Cut multiple images to create one image
    - There is not enough variation in the data, so we need to create more data.
<p align="middle">
  <img src="https://github.com/MinTagg/VISOL_Classifier/assets/98318559/0b7aadcd-018a-4a87-a716-0295f74830c6">
  <img src="https://github.com/MinTagg/VISOL_Classifier/assets/98318559/51693b80-9227-487e-96f4-3b452347def2">
</p>

## Augmentation : Others
  - HSV
  - Rotation
  - Flip
  - Gaussian Noise
  - translation
  - shear

<p align="middle">
  <img src="https://github.com/MinTagg/VISOL_Classifier/assets/98318559/059fa8d3-d861-409a-9ec0-ec7515a7ec17">
  <img src="https://github.com/MinTagg/VISOL_Classifier/assets/98318559/9a4ed712-2c88-49d7-80c9-d6ad4d86cf37">
</p>


