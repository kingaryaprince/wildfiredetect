# Wildfire Image Classification using Deep Learning

This directory contains a collection of machine learning programs, all aimed at image classification, intending to assess and compare their performance. 
The `fire_detect_cnn.py` program utilizes Convolutional Neural Networks (CNNs) to conduct binary classification, determining the presence or absence of wildfires in images.

## Requirements

- Python 3.x
- Libraries:
  - `logging`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `keras`
  - `sklearn`

## Installation

Clone the repository or download the source code. Ensure that you have the required libraries installed. You can install them using pip:

```bash
pip install pandas numpy matplotlib keras sklearn
```

## Usage
The program can be run from the command line with various arguments to specify the data path, number of epochs, batch size, optimizer, and whether to train or evaluate the model.

### Example Command to Train the Model:
```bash
python script_name.py --path "path/to/data" --epochs 10 --batch 32 --optimizer adam --train
```

### Example Command to Evaluate the Model:
```bash
python script_name.py --path "path/to/data" --model_path "path/to/model.keras" --batch 32
```

## Features
**Image Data Processing**: Fetches and splits image data into training, validation, and test sets.
**Model Training: Trains** a CNN model with specified parameters.
**Performance Evaluation**: Evaluates the model on the test set and prints metrics like accuracy, F1 score, precision, recall, and ROC AUC.
**Visualization**: Generates and saves plots showing the model's accuracy and loss over 
