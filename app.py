# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:43:41 2020

@author: Admin 2
"""
import streamlit as st 
import tensorflow as tf
import joblib,os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import re
from nltk.corpus import stopwords


model=tf.keras.models.load_model( 'model.h5' )

st.title("Email Classifier")
	# st.subheader("ML App with Streamlit")
html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Streamlit ML App </h1>
	</div>

	"""
st.markdown(html_temp,unsafe_allow_html=True)
    
news_text = st.text_area("Enter email Here","Type Here")
def clean_text(text):
        text = re.sub('[^a-zA-Z]', ' ', text)  
        text = text.lower()  
        text = text.split(' ')  
        text = [w for w in text if not w in set(stopwords.words('english'))] 
        text = ' '.join(text)    
            
        return text
if st.button("Classify"):
    tokenizer_obj=Tokenizer()
    tokenizer_obj.fit_on_texts(news_text)
    sequences=tokenizer_obj.texts_to_sequences(news_text)
    text_cleaned = clean_text(news_text )
    test_sequences = tokenizer_obj.texts_to_sequences(text_cleaned)
    test_data = pad_sequences(test_sequences, maxlen=100)
    predictions= model.predict_classes(test_data)
   
    # load the model
    st.write(np.argmax( predictions[0]))  
    if( predictions[0]==1):
        st.write('sparm')
    else:
        st.write('Ham')
        






