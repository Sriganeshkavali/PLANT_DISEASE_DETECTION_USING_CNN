import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import json 

# =======================================================================
# --- CONFIGURATION (Common to all components) ---
# =======================================================================

# Paths
MODEL_PATH = 'my_best_plant_model.keras' 
CLASS_NAMES_FILE = 'class_names.json' # File saved by training script

# Model Input Parameters (MUST match the size used during training)
IMAGE_SIZE = (150, 150)

# =======================================================================
# --- UTILS LOGIC (Image Processing and Model/Class Loading) ---
# =======================================================================

@st.cache_resource
def load_and_cache_model(model_path):
    """Loads the Keras model and caches it."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info(f"Please ensure the model file '{MODEL_PATH}' is available.")
        return None

@st.cache_resource
def load_class_names(names_file):
    """Loads the ordered list of class names from the JSON file."""
    try:
        with open(names_file, 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        st.error(f"Class names file not found: {names_file}")
        st.info("Please run your training script ('plant_disease_classifier.py') first to generate this file.")
        return None
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return None

def preprocess_image(uploaded_file):
    """Loads, resizes, and normalizes the image for model input."""
    # 1. Load the image from the uploaded file buffer
    img = Image.open(io.BytesIO(uploaded_file.read()))
    
    # 2. Resize (MUST match training size)
    img_resized = img.resize(IMAGE_SIZE)
    
    # 3. Convert to array and normalize
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    
    # 4. Expand dimensions to create a batch of size 1: (1, 150, 150, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img, img_batch

# =======================================================================
# --- PREDICTION LOGIC (Model Inference) ---
# =======================================================================

def predict_class(model, class_names, img_batch):
    """Performs the prediction and extracts the result."""
    
    # 1. Get predictions (returns array of probabilities)
    predictions = model.predict(img_batch, verbose=0)
    
    # 2. Find the index of the highest probability
    predicted_index = np.argmax(predictions[0])
    
    # 3. Get the class name and confidence
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    
    return predicted_class, confidence, predictions[0]

# =======================================================================
# --- STREAMLIT UI (APP LOGIC) ---
# =======================================================================

def main():
    st.set_page_config(page_title="Plant Disease Predictor", layout="centered")

    st.title("🌱 Automated Plant Disease Predictor")
    st.markdown("""
        Upload a leaf image to get an immediate prediction on its health status 
        and potential disease.
    """)

    # 1. Load the model and class names once
    model = load_and_cache_model(MODEL_PATH)
    class_names = load_class_names(CLASS_NAMES_FILE)

    if model is None or class_names is None:
        return # Stop execution if model or class names failed to load

    # 2. File Uploader
    uploaded_file = st.file_uploader(
        "Choose an image of a plant leaf...",
        type=['jpg', 'jpeg', 'png', 'webp']
    )

    if uploaded_file is not None:
        # Display loading spinner while processing
        with st.spinner('Processing image and running prediction...'):
            
            # --- Call Utils Logic ---
            original_image, processed_batch = preprocess_image(uploaded_file)
            
            # --- Display Image ---
            st.image(original_image, caption='Uploaded Image', use_column_width=True)

            # --- Call Prediction Logic ---
            predicted_class, confidence, probabilities = predict_class(model, class_names, processed_batch)

        # 3. Display Results
        st.success("✅ Prediction Complete!")
        
        col1, col2 = st.columns([1, 2])
        
        # Clean up the class name for display (e.g., Apple_scab -> Apple scab)
        display_class = predicted_class.replace('_', ' ').replace('___', ' - ')

        with col1:
            st.metric(
                label="Predicted Class", 
                value=display_class, 
                delta=f"{confidence:.2f}% Confidence"
            )
        
        with col2:
            st.markdown("### Confidence Scores")
            
            # Display top 5 predictions in a bar chart
            top_indices = np.argsort(probabilities)[::-1][:5]
            top_classes = [class_names[i].replace('_', ' ').replace('___', ' - ') for i in top_indices]
            top_scores = [probabilities[i] * 100 for i in top_indices]
            
            # Create a dictionary for the bar chart data
            chart_data = {'Class': top_classes, 'Confidence (%)': top_scores}
            st.bar_chart(chart_data, x='Class', y='Confidence (%)')
            
    else:
        st.info("Awaiting image upload...")

if __name__ == '__main__':
    # Initialize the app
    main()
