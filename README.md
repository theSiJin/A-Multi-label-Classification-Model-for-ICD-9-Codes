This repo contains code and data files prepared for CSE 6250 final project.



## Data

The MIMIC-III dataset is not included. Please refer to its official website for access. 

`HADM_IDS` and code descriptions are already included in the `data` folder.



## Environment

Python and PySpark are necessary to run the program. All codes are tested in local enviornment with Python 3.8.



To set up PySpark on local mode and work with Jupyter Notebook, please refer to some online tutorials (for example, [this one](https://opensource.com/article/18/11/pyspark-jupyter-notebook) on opensource.com).



Necessary packages are listed in `requirements.txt`. Use `pip` to install them from the file.

## Execution

To start from sratch, execute the following steps.

1. ETL

   Follow steps in `etl.ipynb` to load and clean data from MIMIC-III dataset. The program will save the output in the `data` folder.

2. Data Preprocessing

   Follow steps in `preprocess_text.ipynb` to make preparations for the model training.

3. Training

   Adjust the paramters in `learn.py` and run this program to train the model and get the evaluation results. Models after training will be save in `model` folder for further study.

## Credit

The codes of regularization module are inspired and improved from CAML ([github](https://github.com/jamesmullenbach/caml-mimic)), which is also the paper that I tried ti replicate in the proposal.