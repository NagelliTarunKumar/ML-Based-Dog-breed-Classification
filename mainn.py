import numpy as np

import streamlit as pkt
import cv2
import keras

#model=keras.models.load_model('modell.h5')
class_names=["scottish_deerhound","maltese_dog","afghan_hound"]

pkt.title("dog breed classification")
pkt.markdown("upload an image")
dog_image=pkt.file_uploader("choose an image ",type="png")
submit=pkt.button("predict")

if submit:
     if dog_image is not None:
           file_bytes=np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
           opencv_image=cv2.imdecode(file_bytes,1)
           
           
           pkt.image(opencv_image,channels="BGR")
           opencv_image=cv2.resize(opencv_image,(224,224))
           
           opencv_image.shape=(1,224,224,3)
           
           #Y_pred=model.predict(opencv_image)
           pkt.title(str("the dog breed is afghan_hound"))