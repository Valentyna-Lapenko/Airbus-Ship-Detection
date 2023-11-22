# Semantic segmentation model
## Files
Airbus Ship Detection EDA.ipynb - Jupyter notebook with exploratory data analysis

requirements.txt - txt file with the list of required python modules.

Unet.py - code for building the Unet architecture.

model_training.py - source code for model training.

model_testing.py - source code for model testing.
## Description of the Dataset
Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

For this metric, object segments cannot overlap. There were a small percentage of images in both the Train and Test set that had slight overlap of object segments when ships were directly next to each other. Any segments overlaps were removed by setting them to background (i.e., non-ship) encoding. Therefore, some images have a ground truth may be an aligned bounding box with some pixels removed from an edge of the segment. These small adjustments will have a minimal impact on scoring, since the scoring evaluates over increasing overlap thresholds.

The train_ship_segmentations.csv file provides the ground truth (in run-length encoding format) for the training images.

The sample_submission files contain the images in the test images.

## Model
The architecture used is U-Net, which is very common for image segmentation problems.

Evaluation metric is Dice Sore.

Training of the model was done using Adam optimizer with learning rate 1.e-3.

The model is trained for 3 epochs.

### Links
https://www.kaggle.com/competitions/airbus-ship-detection/data
