import tensorflow as tf
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from keras_preprocessing.sequence import pad_sequences
import pickle

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
model = tf.keras.models.load_model('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/LSTM_model', compile=False)

print(model.summary())

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)


q1 = input('Enter first question: ')
q2 = input('Enter second question: ')

t1 = text_to_wordlist(q1)
t2 = text_to_wordlist(q2)


tokenizer = Tokenizer(num_words=200000)
tokenizer = pickle.load(open('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/tokenizer.pickle','rb'))

s1 = tokenizer.texts_to_sequences(t1)
s2 = tokenizer.texts_to_sequences(t2)
word_index = tokenizer.word_index

data_1 = pad_sequences(s1, maxlen=30)
data_2 = pad_sequences(s2, maxlen=30)

nb_words = min(200000, len(word_index))+1

sample_1 = np.vstack((data_1, data_2))
sample_2 = np.vstack((data_2, data_1))

pred = model.predict([sample_1, sample_2], verbose = 1)
pred += model.predict([sample_2, sample_1], verbose = 1)
pred /= 2
print([])
print(pred.ravel())