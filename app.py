import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("fruit_disease_model.keras")

# Class names (MUST match training order)
class_names = [
    "apple_black_rot",
    "apple_blotch",
    "apple_healthy",
    "apple_scab",
    "mango_alternaria",
    "mango_anthracnose",
    "mango_black_mold",
    "mango_healthy",
    "mango_stem_rot"
]

st.title("Fruit Disease Detection System")
st.write("Upload a clear image of an Apple or Mango fruit")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Convert image to RGB and resize
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    st.image(image, caption="Uploaded Image", width=300)



    # Preprocess image
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # Display result
    st.success(f"Predicted Disease: {class_names[predicted_class]}")
    st.info(f"Prediction Confidence: {confidence:.2f}%")
