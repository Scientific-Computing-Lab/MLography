# MLography: An Machine-Learning approach for Metallography

Anomaly Detection for impurities: There are several kinds of anomaly measures:
1. **Spatial Anomaly**: impurities that are big and distant compared to their neighborhood are considered anomalous.
2. **Shape Anomaly**: impurities of an non-symmetric shapes are considerd anomalous.
2. **Area Anomaly**: locating and quantifying areas of anomalous impurities.

# Citation

If you found these codes useful for your research, please consider citing MLography

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
python anomaly_detection.py --input_scans=<input directory of scans, we used "./tags_png_cropped/\*"> --model_name="\<auto-encoder-model-name\>" --min_threshold=<used for pre-processing, we used 30> --area_anomaly_dir=<log direcory for output, default is "./logs/area/">

## Training

In order to train the auto-encoder model for the shape anomaly measure, use:

python neural_net.py --model_name="\<model name without file extension\>" --anomaly_blank_label=\<True if the use of blank labels for anomalous objects is desired\>

# Data-set
The data-set can be found at *tags_png_cropped* directory
