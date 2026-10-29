import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Skin Cancer Detection App")
st.write("Apni skin ki image upload karein taake model check kar sakay.")

# Model load karna (.h5 file github par honi chahiye)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('skin_cancer_detection_model.h5')
    return model

try:
    model = load_model()
except Exception as e:
    st.warning("Model file nahi mili! Baraye meharbani 'skin_cancer_detection_model.h5' file GitHub par upload karein.")
    model = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    if model is not None:
        st.write("Classification chal rahi hai...")
        
        # Image ko 128x128 resize karna (jaisa model train karte waqt kiya tha)
        image = image.resize((128, 128))
        img_array = np.array(image)
        
        # Agar image mein 4 channels (RGBA) hon toh usay 3 channels (RGB) mein convert karna
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction lena
        predictions = model.predict(img_array)
        
        # Classes (Benign vs Malignant)
        class_names = ['Benign (Non-Cancerous)', 'Malignant (Cancerous)']
        
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)
        
        st.success(f"Result: **{predicted_class}** (Confidence: {confidence:.2f}%)")
    else:
        st.error("Model load nahi ho saka, is liye prediction nahi ki ja sakti.")
