import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow import keras 
from keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from keras.preprocessing import image 

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            body {
            background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
            background-size: cover;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = "cnn_pneu_vamp_model.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()
st.write("""
# X-Ray Classification [Pneumonia/Normal]
""")



  


temp = st.file_uploader("Upload X-Ray Image")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(tf.keras.utils.load_img(temp_file.name))


if buffer is None:
  st.info("Oops! that doesn't look like an image. Try again.")

else:

 

  hardik_img = tf.keras.utils.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')

  # Preprocessing the image
  pp_hardik_img = tf.keras.utils.img_to_array(hardik_img)
  pp_hardik_img = pp_hardik_img/255
  pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)

  #predict
  hardik_preds= cnn.predict(pp_hardik_img)
  if hardik_preds>= 0.5:
    out = ('The model is {:.2%} percent confirmed that this is a Pneumonia case'.format(hardik_preds[0][0]))
  
  else: 
    out = ('The model is {:.2%} percent confirmed that this is a Normal case'.format(1-hardik_preds[0][0]))

  st.success(out)
  
  image = Image.open(temp)
  st.image(image,use_column_width=True)
         