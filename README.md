# Supreme Court Case Outcome Prediction

This project implements a Bayesian Network to predict Supreme Court case outcomes, as described in the paper "Supreme Court Case Outcome Prediction Using Bayesian Networks".

## How to Use

### 1. Setup
Create a Python environment and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Get the Data
Data can be found in the "SCDB_2025_01_justiceCentered_Citation" zip folder as a csv file

### 3. (First Time Only) Build the CPTs

Before you can run predictions, you must process the CSV and build the CPT file.

Important: Open build_cpts.py and edit the NETWORK_STRUCTURE list. You must replace the placeholder variable names with the actual column names from the CSV file.

Run the build script:
```bash
python build_cpts.py
```
This will create a file named scdb_cpts.pkl

### 4. Run a Prediction
Now you can run the main inference script.
1. Open main.py.
2. Change the observations dictionary to include the case attributes you want to test.
3. Change the target_variable and target_value to query the probability you're interested in.

Run the file:
```bash
python main.py
```
