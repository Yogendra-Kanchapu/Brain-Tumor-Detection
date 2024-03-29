import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('./cnn_model.h5')

# Function to preprocess the input images
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize the image to match model's expected input
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    image = tf.constant(image, dtype=tf.float32)
    return image

# Streamlit UI
def main():
    st.title("Brain Tumor Prediction")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        # Preprocess and predict
        processed_image = preprocess_image(uploaded_file)
        prediction = model.predict(processed_image)

        # Display the prediction
        if prediction[0][0] > 0.5:  # Assuming your model outputs a probability
            st.write('Prediction: Brain Tumor detected')
        else:
            st.write('Prediction: No Brain Tumor detected')




