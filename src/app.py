import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('../models/image-classifier.keras')

st.title("It's a CAT or a DOG?")
st.markdown("""Power by: [Adam Candalija Naranjo](https://chocobar.net)""")
st.divider()

val1 = st.file_uploader('Upload your CAT or Dog Image', type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if val1 is None:
        st.warning("Por favor sube una imagen antes de predecir.")
    else:
        image = Image.open(val1).resize((128, 128))
        image = np.asarray(image)
        image = image.astype('float32') / 255.0
        st.image(image)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)[0][0]
        st.divider()
        if pred > 0.7:
            st.write("What a beautyfil cat")
        elif pred > 0.5:
            st.write("I'm pretty sure that's a cat")
        elif pred > 0.3:
            st.write("I'm pretty sure that's a dog")
        else:
            st.write("What a beautyfil dog")
        st.divider()
        st.image(image)

