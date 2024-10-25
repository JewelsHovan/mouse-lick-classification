# READ_ME_DOC

The following data and code are provided for:

**Machine learning-based automated phenotyping of inflammatory nocifensive behavior in mice**  
by Janine M. Wotton, Emma Peterson, Laura Anderson, Stephen A. Murray, Robert E. Braun, Elissa J. Chesler,  
Jacqueline K. White, and Vivek Kumar - *in Molecular Pain, October 2020*

## DeepLabCut

1) **labels_images_DLC_data.7z**  
   Zipped file containing the images used to train DeepLabCut and the associated human-annotated labels in an excel spreadsheet.

2) **train_cfg.yaml**  
   The yaml output from DLC after training completed.

3) **test_cfg.yaml**  
   The test yaml output from DLC after training completed.

4) **LabeledImages_DeepCut_resnet_50_85shuffle1_750000forTask_footshuffle_miss.7z**  
   The images with labels from human and model superimposed for train and test images.

## GentleBoost classifier

5) **train40.7z**  
   The output of DLC for each of these 40 training videos in h5 files.

6) **training_labels.7z**  
   The classification of one arena for each of the 40 videos. Arena indicated by file name.

7) **trainedModelclassifier.mat**  
   Matlab structure of the GentleBoost classifier.

8) **classifier_variables.m**  
   Matlab code to create input to the classifier.