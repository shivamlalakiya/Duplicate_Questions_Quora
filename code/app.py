import streamlit as st
from prep import preprocess
import tensorflow as tf
import os
import re
import pickle
from prep import preprocess
import numpy as np
from keras_preprocessing.sequence import pad_sequences

st.set_page_config(page_title='Duplicate Questions')
                   
st.title("Quora Question Pairs - Predicting Duplicate Questions")

st.write("Welcome to my Quora Question Pairs model implementation page! As a data scientist,\
         I have developed a machine learning model that predicts whether a pair of questions from\
         the Quora Question Pairs dataset have the same meaning. I used a combination of Natural Language \
        Processing (NLP) techniques and machine learning algorithms to preprocess and model the data.")


q1 = st.text_input('The First Question', '')
q2 = st.text_input('The Second Question', '')
button_check = st.button("Submit")
if q1 and q2 and button_check:
    p = preprocess()
    t1 = []
    t2 = []
    t1.append(p.text_to_worldlist(q1))
    t2.append(p.text_to_worldlist(q2))


    tokenizer = pickle.load(open('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/tokenizer.pickle','rb'))

    s1 = tokenizer.texts_to_sequences(t1)
    s2 = tokenizer.texts_to_sequences(t2)

    data_1 = pad_sequences(s1, maxlen=30)
    data_2 = pad_sequences(s2, maxlen=30)


    sample_1 = np.vstack((data_1, data_2))
    sample_2 = np.vstack((data_2, data_1))

    model = tf.keras.models.load_model('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/LSTM_model', compile=False)
    pred = model.predict([sample_1, sample_2], verbose = 1)
    pred += model.predict([sample_2, sample_1], verbose = 1)
    pred /= 2
    pr = pred.ravel()[0]
    similar_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">The questions are similar</p>'
    nonsimilar_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">The questions are not similar (Differnet)</p>'

    if pr > 0.1:
        st.markdown(similar_title, unsafe_allow_html=True)
    else:
        st.markdown(nonsimilar_title, unsafe_allow_html=True)


st.header('Model Architecture')

st.write("The model architecture consists of three main parts: word embeddings, recurrent neural networks (RNNs), \
         and a binary classifier. Word embeddings are used to represent the meaning of words as dense vectors. \
         I used the Word2Vec algorithm to learn word embeddings from the training set of the Quora Question Pairs dataset.")

st.write("Next, I used a type of RNN called Long Short-Term Memory (LSTM) to model the sequence of words in each question pair.\
          LSTMs help model long-term dependencies in sequential data like natural language. I used a bi-directional LSTM to encode\
          the sequence of word embeddings from both questions in a pair..")

st.write("Finally, the encoded question pairs are fed into a binary classifier to predict whether the questions have the same meaning. \
         I used a simple feedforward neural network with a single sigmoid output neuron for this task.")


st.header('Model Training and Evaluation')
st.write("To train and evaluate the model, I used the log loss metric. I trained the model on the training set of the Quora Question Pairs\
          dataset using a batch size of 2048 and an Nadam optimizer. I used early stopping and model checkpointing to prevent overfitting and to save the best model based on validation loss.")


st.header('Implementation')
st.write("To implement the model, input two questions you want to compare for meaning. The model will output a probability score between 0 and 1,\
          where values closer to 1 indicates that the two questions have the same meaning.")

st.header('Conclusion')
st.write("I hope my model implementation can be helpful to anyone interested in NLP and machine learning, and I welcome any feedback or suggestions. \
         Please try out my model and let me know what you think!")



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def footer():
    myargs = [
        "Made with ❤️ by Shivam",
    ]
st.write( "Made with ❤️ by Shivam")
if __name__ == "__main__":
    footer()

