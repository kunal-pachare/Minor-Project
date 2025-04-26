import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('banana_carbide_detector (1).h5')  # <-- replace with your model filename

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((128, 128))  # <-- resize to match model input size (adjust if needed)
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Streamlit UI
st.title("Banana Ripeness Prediction App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # if model gives class probabilities

        # Show result
        if predicted_class == 1:
            st.success("✅ The banana is **Naturally Ripened**! i.e Without carbide")
        else:
            st.warning("⚠️ The banana is **Chemically Ripened**! i.e With")

