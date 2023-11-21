# Semantic segmentation model
## Files

## Description of the Dataset
Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

For this metric, object segments cannot overlap. There were a small percentage of images in both the Train and Test set that had slight overlap of object segments when ships were directly next to each other. Any segments overlaps were removed by setting them to background (i.e., non-ship) encoding. Therefore, some images have a ground truth may be an aligned bounding box with some pixels removed from an edge of the segment. These small adjustments will have a minimal impact on scoring, since the scoring evaluates over increasing overlap thresholds.

The train_ship_segmentations.csv file provides the ground truth (in run-length encoding format) for the training images.

The sample_submission files contains the images in the test images.

## Model
The architecture used is U-Net, which is very common for image segmentation problems.

Evaluation metric : Dice score is Dice Sore.

Training of the model was done using Adam optimizer with learning rate 1.e-4.

The model is trained for 3 (3500 steps on each) epochs.

After 3 epochs we get Dice Score - 0.602.

### Links
https://www.kaggle.com/competitions/airbus-ship-detection/data
