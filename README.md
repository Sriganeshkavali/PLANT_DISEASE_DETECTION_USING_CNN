# Plant Disease Classifier (CNN)

## Dataset Link : 
* https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
## Project Overview
* This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify plant leaf images into various categories, identifying specific diseases or healthy states. The model is trained on an augmented dataset and deployed via a user-friendly web interface built with Streamlit.

## Getting Started
* Follow these steps to set up your environment, train the model, and run the prediction application.

## Prerequisites
* You need Python 3.8+ installed. All required packages are listed in requirements.txt.

## Install Dependencies:
*  Create a Virtual Environment (Recommended)
* pip install -r requirements.txt



## Data Setup
* The project assumes the following directory structure for the training data, based on the path provided:

.
└── New Plant Diseases Dataset(Augmented)/
    └── New Plant Diseases Dataset(Augmented)/
        └── train/
            ├── Apple___Apple_scab/
            ├── Corn___Cercospora_leaf_spot/
            ├── ... (45 class folders)

The scripts use the base path defined inside the code: BASE_PATH = "New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)".

## Model Training
* The training script performs EDA, loads and augments data, builds a deep CNN, and saves the resulting model and class list.

1. Run the Training Script
Execute the training script. This process will:

Perform Exploratory Data Analysis (EDA) and display distribution plots.

Build and train the CNN model with Early Stopping.

Save the final model weights in the native Keras format.

CRITICALLY: Save the ordered list of class names to class_names.json.

python plant_disease_classifier.py

Generated Artifacts
Upon successful completion, the script will create two files in the root directory:

my_best_plant_model.keras (The trained CNN model weights and architecture).

class_names.json (The ordered list of class names required by the Streamlit app).

## Deployment (Streamlit App)
The prediction app allows users to upload an image and receive a real-time diagnosis.

 Run the Streamlit App
Launch the user interface from your terminal:

streamlit run app.py

The application will open in your browser, allowing you to upload an image for prediction.

Prediction Output
The application displays:

The Uploaded Image.

The Predicted Class (e.g., Tomato Healthy).

The Confidence Score for the top prediction.

A Confidence Scores Chart showing the top 5 predicted classes.
