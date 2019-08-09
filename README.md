# MLography: An Machine-Learning approach for Metallography

Currently the first phase is under development: Anomaly Detection for impurities. There are several kinds of anomalies:
1. **Spatial Anomaly**: impurities that are big and distant compared to their neighborhood are considered as anomalies.
2. **Self Anomaly**: impurities of an non-symmetric shapes are considerd as anomalies.

# Instructions
## Running

There are several scripts in the project:
1. **impurity_extract.py** - the main script of the first phase. Currently it allows to execute the *Spatial Anomaly* functionality (*weighted_kth_nn*) as well as to supply an infrastructure for writing the impurities (*normalize_circle_boxes*) after pre-processing (*get_markers*) into files. 
2. **neural_net.py** - the network model that is responsible for training and loading data for the *Self Anoamly*.
3. **test_model.py** - a test for the neural network above.

## Data
All data can be found in the directory *data/* . The seperation into two classes that we used for the Neural Network training and validation can be found in the directory *data/data_two_classes*.
