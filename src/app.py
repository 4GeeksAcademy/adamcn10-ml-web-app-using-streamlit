import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('/workspaces/adamcn10-ml-web-app-using-streamlit/models/image-classifier.keras')

st.title("It's a CAT or a DOG?")
st.markdown("""Power by: [Adam Candalija Naranjo](https://chocobar.net)""")
st.divider()

val1 = st.file_uploader('Sube tu imagen de gato o perro', type=["jpg", "jpeg", "png"])

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
            st.write("Que gato más bonito")
        elif pred > 0.5:
            st.write("Casi seguro que eso es un gato")
        elif pred > 0.3:
            st.write("Çasi seguro que eso es un perro")
        else:
            st.write("Que perro más bonito")
        st.divider()
        st.image(image)

