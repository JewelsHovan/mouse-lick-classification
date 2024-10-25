
## 1. Successfully downloaded the data

This is the folder structure I went with
```
- Data
    - DeepLabCut
        - LabeledImages_DeepCut_resnet_50_85shuffle1
        - labels_images_DLC_data
        - test_cfg.yaml
        - train_cfg.yaml
    - GentleBoost
        - train40
        - training_labels
```

## 2. We matched all training and label files and combined them into a single numpy array.

The data is stored in h5 files. To read them, we first examine the keys to isolate the dataset. Once we understand the schema of one file, we can read the entire dataset.

A key issue was the mismatch in file naming conventions between the training data and label files. The training data files included the date in their filenames, while the training label files did not. The filenames for training data included ARENA, Tetrad, RUN, and DATE, whereas the label files included ARENA, Tetrad, RUN, DATE, and SUFFIX.

Multiple runs with the same TETRAD/RUN/ARENA but different DATEs were labeled with a suffix. If a file had no suffix, it indicated the earliest date. Files with suffixes (e.g., 'a', 'b') represented later dates sequentially.

For example, we renamed 
```
'Arena1-Run1 Tetrad 2 8_20_2018 9_10_26 AM 2DeepCut_resnet50_footshuffle_missFeb24shuffle1_750000.h5' to 'inpA1R1T2labels.h5' in 'train40/renamed_files'.
```

**NOTE**: We ignored the 36th index due to a row mismatch by 1, which appears to be a data entry error that needs manual verification.

## 3. Once we have the single numpy array, we can split it into the training and testing data

We used `train_test_split` from `sklearn.model_selection` to divide the data into training and testing sets. The data was standardized using `StandardScaler` to ensure that each feature contributes equally to the model's performance.

## 4. Training some Classifiers

We started by training a dummy classifier using the `most_frequent` strategy to establish a baseline. The dummy classifier achieved an accuracy of approximately 83%, highlighting the imbalance in the dataset.

Next, we trained a Logistic Regression model, which improved the accuracy to around 86%. The model was trained with a maximum of 1000 iterations to ensure convergence.

We also experimented with a RandomForestClassifier, which provided a further boost in accuracy to approximately 89.5%. This model used 200 estimators with a maximum depth of 5.

Finally, we employed XGBoost, a powerful ensemble model, which achieved an impressive accuracy of 98%. This model was configured with 250 estimators and a maximum depth of 4, demonstrating its effectiveness on the imbalanced dataset.

## 5. Future Steps

- Investigate the data entry error at index 36 to resolve the mismatch in rows.
- Explore additional models and hyperparameter tuning to further improve performance.
- Consider techniques to address data imbalance, such as resampling or using specialized algorithms.



