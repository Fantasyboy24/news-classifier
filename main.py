import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model('news-classification-model.h5')
data = pd.read_csv('ag-news-classification-dataset/train.csv')

# Load tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['Title'] + ' ' + data['Description'])

# Define function to make prediction
def predict_class(title, description):
    # Tokenize input
    text = title + ' ' + description
    x = tokenizer.texts_to_sequences([text])
    x = pad_sequences(x, maxlen=100, padding='post', truncating='post')
    # Make prediction
    y_pred = model.predict(x)
    # Decode prediction
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    class_id = y_pred.argmax(axis=-1)[0]
    class_name = class_names[class_id]
    return class_name

# Define Streamlit app
st.title('News Article Classifier')
title = st.text_input('Title')
description = st.text_area('Description')
if st.button('Classify'):
    class_name = predict_class(title, description)
    st.write('Predicted class:', class_name)