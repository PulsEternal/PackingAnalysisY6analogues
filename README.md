Dataset and feature extracting code for manuscript

# Concretized Structural Evolution Supported Assembly-Controlled Film-Forming Kinetics in Slot-die Coated Organic Photovoltaics

**Please cite the original work if you intent to used either the dataset or code in this repository.**  



## contents

This repository contains:

* Explanatory file (this one): README.md 

* Dataset used to train the packing style classification model for Y6 analogues
  * dataset_X.npy - feature vector for extracted packing styles
  * dataset_y.npy - tagged packing information for packing styles in dataset_X
* code used to the generate the feature vector : calc_feature_vec.py
* An example how to use the ML model to predict the packing style: example.py
* An reference molecule structural file of Y6 molecule, for atom index reference in calculating feature vector : y6_ref.sdf.
* Visualized examples of classifications, under visual classifications directory.



## requirements

* python (of course)
* numpy package
* scikit-learn package



## recommendations 

* One could integrate the code and dataset with Jupiter Notebooks for better experience

* suggested visualization package:
  Atomic Simulation Environment
  [ASE]:https://wiki.fysik.dtu.dk/ase/#

