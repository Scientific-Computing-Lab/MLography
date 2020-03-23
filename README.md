# MLography: An Automated Quantitative Metallography Model for Impurities Anomaly Detection using Novel Data Mining and Deep Learning Approach


# Introduction
This repository is the implementation of MLography - a framework that measures and marks the anomaly scores of some geometrical objects. In the following output examples of metallographic scans, red objects (impurities) are the most anomalous and blue are the most non-anomalous.
## Output examples ##
In MLography there are several kinds of anomaly measures:
1. **Spatial Anomaly**: objects (impurities) that are big and distant compared to their neighborhood are considered anomalous:

![Spatial Anomaly on a sample image](https://github.com/matanr/MLography/blob/master/spatial.PNG)
2. **Shape Anomaly**: objects (impurities) of an non-symmetric shapes are considerd anomalous.

![Shape Anomaly on a sample image](https://github.com/matanr/MLography/blob/master/Shape_anomaly.png)

2.5. **Spatial and Shape Anomaly**: Combining the scores of Spatial and Shape anomalies highlights the most anomalous objects from both measures.
![Spatial and Shape Anomaly combined](https://github.com/matanr/MLography/blob/master/k_%3D_50%2C_Shape_and_Spatial_anomalies_combined.png)
3. **Area Anomaly**: locating and quantifying areas of anomalous objects (impurities).

![Area Anomaly on a sample image](https://github.com/matanr/MLography/blob/master/scan1tag-47.png)

# Citation
For more information about the measures and their means of the implementations, please refer to the paper.
If you found these codes useful for your research, please consider citing MLography.

# Instructions
## Running

There are several scripts:
1. **anomaly_detection.py** - the main script. Currently it allows to execute the *Shape and Spatial Anomaly* functionality (*extract_impurities_and_detect_shape_spatial_anomaly*) and *Area Anomaly* functionality (*extract_impurities_and_detect_anomaly*) which uses the previous measures to locate and quantify the anomalous areas in the scan.
1. **impurity_extract.py** - pre-processing the input scan image (using image processing techniqes such as water-shed algorithm). 
2. **spatial_anomaly.py** - implements the *Spatial Anoamly* functionality, mainly with the Weighted-kth-Neighbour algorithm.
2. **shape_anomaly.py** - pre-step to *Shape Anoamly* functionality, it mainly calculates the difference between areas of each impurity to its minimal enclosing circle. It is used for creating the training set to the auto-encoder model described next.
2. **neural_net.py** - the auto-encoder neural network model that is responsible for training and loading data for the *Shape Anoamly*.
3. **use_model.py** - uses the neural network for prediction and evaluating the reconstruction loss, ass well as testing for the *Shape Anoamly*.
2. **area_anomaly.py** - implements the *Area Anoamly* functionality, mainly with the Market-Clustering algorithm.

To run the program (on a trained auto-encoder model) use:
```
python anomaly_detection.py --input_scans=<input directory of scans, we used "./tags_png_cropped/*"> --model_name="<auto-encoder-model-name>" --min_threshold=<used for pre-processing, we used 30> --area_anomaly_dir=<log direcory for output, default is "./logs/area/">
```

In order to order all the area anomaly add the flag *--order* and if you want to print the precentiles in which all areas of the input scans are placed, add the flag *--print_order*.

## Training

In order to train the auto-encoder model for the shape anomaly measure, on your data use:
```
python neural_net.py --model_name="<model name without file extension>" --anomaly_blank_label=<True if the use of blank labels for anomalous objects is desired>
```

Your data should reside in a directory in data/, then divided to two directories: train/ and validation/, in each one will be one directory - normal/, or two directories - anomaly/ and normal/ if the use of blank labels for anomalous objects is desired. These directories should hold all your data.

For splitting the data to the needed directories use the *split_data.py* script:
```
python anomaly_detection.py --detect=False --order=False --print_order=False prepare_data=True prepare_data_path="<path to data to be rescaled and prepared>"
```


# Data-set
The data-set that was used in the paper can be found in *tags_png_cropped/* directory.
