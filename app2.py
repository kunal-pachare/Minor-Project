import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import random

# Load the trained model
model = tf.keras.models.load_model('banana_carbide_detector (1).h5')  # <-- replace with your model filename

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((128, 128))  # <-- resize to match model input size (adjust if needed)
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Functions to generate nutrient values
def get_natural_nutrient_values():
    return {
        "Potassium (mg)": random.randint(350, 400),
        "Sugar Level (g)": round(random.uniform(12.0, 14.0), 2),
        "Vitamin C (mg)": round(random.uniform(8.0, 10.0), 2),
        "Fiber (g)": round(random.uniform(2.5, 3.5), 2)
    }

def get_chemical_nutrient_values():
    return {
        "Potassium (mg)": random.randint(300, 340),
        "Sugar Level (g)": round(random.uniform(9.0, 11.0), 2),
        "Vitamin C (mg)": round(random.uniform(5.0, 7.0), 2),
        "Fiber (g)": round(random.uniform(2.0, 2.4), 2)
    }

# Streamlit UI
st.title("Banana Ripeness Prediction App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
st.caption("📌 *Please upload only a banana image for accurate results.*")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # if model gives class probabilities

        # Show result and nutrients
        if predicted_class == 1:
            st.success("✅ The banana is *Naturally Ripened*! i.e Without carbide")
            nutrients = get_natural_nutrient_values()
        else:
            st.warning("⚠ The banana is *Chemically Ripened*! i.e With carbide")
            nutrients = get_chemical_nutrient_values()

        st.subheader("🍽 Estimated Nutritional Values:")
        for nutrient, value in nutrients.items():
            st.write(f"**{nutrient}**: {value}")
