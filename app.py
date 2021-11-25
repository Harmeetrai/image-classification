import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import load_model
import numpy as np
import shutil

import os # inbuilt module
import random # inbuilt module
import webbrowser # inbuilt module

#================================= About =================================
st.write("""
##  About
	""")
image = Image.open('image-classification.png')
st.image(image, caption='Image Classification', use_column_width=True)
st.write("""
Welcome to this project. This is a Image Classification Project using Tensorflow 2.0 and Streamlit.
	""")
st.write("""
This app can predict the following classes:
        
        1: 'automobile'
        2: 'bird'
        3: 'cat'
        4: 'deer'
        5: 'dog'
        6: 'frog'
        7: 'horse'
        8: 'ship'
        9: 'truck'
        10: 'airplane'
	""")
st.write("""
You have to upload your own test images to test it!
	""")

#========================== File Uploader ===================================
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")

try:
	image = Image.open(img_file_buffer)
	img_array = np.array(image)
	st.write("""
		Preview 👀 Of Given Image!
		""")
	if image is not None:
	    st.image(
	        image,
	        use_column_width=True
	    )
	st.write("""
		**Click The '👉🏼 Predict' Button To See The Prediction Corresponding To This Image! **
		""")
except:
	st.write("""
		### No Picture hasn't selected yet!
		""")

#================================= Predict Button ============================
st.text("""""")
submit = st.button("👉🏼 Predict")

#=========================== Predict Button Clicked ==========================
if submit:

    # save image on that directory
    save_img("test_image.png", img_array)

    image_path = "test_image.png"
    # Predicting
    st.write("👁️ Predicting...")

    loaded_model = load_model("cifar10_model.h5")

    results = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    img = load_img(image_path, 
                   target_size=(224, 224))
    img = img.resize((32, 32))
    img = np.array(img)
    img = img.astype('float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = loaded_model.predict(img)
    st.write("Prediction: \n", results[np.argmax(pred)])


st.write("""
Please check out my github repo for more projects: https://github.com/Harmeetrai\n
Also check out my blog: https://harmeetrai.me
    """)