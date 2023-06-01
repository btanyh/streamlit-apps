# importing necessary modules
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model # to load model from saved file earlier
from tensorflow.keras.preprocessing import image # to do image processing in the required format for predictions
st.set_page_config(layout="wide",page_title = 'Skin Lesion Classification App')# setting streamlit to use the full page, and to set the title of the page
#to remove watermark and menu bar
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

@st.cache_resource # to cache models so it doesn't reload everytime.
def load_model():
  model=tf.keras.models.load_model('./benign_malignant_final_model.h5')# loading in the trained model
  return model
with st.spinner('Model is being loaded..'):# to tell users model is loading
  model=load_model()

st.title('Skin Lesion Classifier')# title of the app shown to users


file = st.file_uploader("Please upload an image of a skin lesion in either jpg or png extension.", type=["jpg", "png"])# user instructions

if file is None:
    st.write(f'<p style="font-size:26px;color:black;">Please upload an image file to be classified as benign or malignant.</p>', unsafe_allow_html=True) # user instructions
### if uploaded file is an image, run the following lines of code
else:
    test_image = image.load_img(file,target_size=(224,224))#resize image
    st.image(test_image)#display image
    # step 4: Convert the image to a matrix of numbers to feed into model
    test_image = image.img_to_array(test_image) # 1st: convert loaded image to array
    test_image = np.expand_dims(test_image, axis=0) # 2nd: https://www.tensorflow.org/api_docs/python/tf/expand_dims (to add additional 4th dummy dimension for batch on top of height, width, channel for a color image, to meet Tensorflow's expected no. of dimensions for input image
    result = model.predict(test_image)# predict the probability of the image
    for pred in result:
        if pred[0] > 0.5:
            text = 'Your skin lesion has been classified as malignant melanoma. Please visit a doctor IMMEDIATELY!'
            st.write(f'<p style="font-size:26px;color:red;">{text}</p>', unsafe_allow_html=True)
        else:
            text = 'Your skin lesion has been classified as benign. Nothing to worry about.'
            st.write(f'<p style="font-size:26px;color:green;">{text}</p>', unsafe_allow_html=True)
    
