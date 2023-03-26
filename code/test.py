import tensorflow as tf
import os
import re
import pickle
from prep import preprocess
import numpy as np
from keras_preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/LSTM_model', compile=False)

# q1 = input('Enter first question: ')
# q2 = input('Enter second question: ')

q1 = "How does the Surface Pro himself 4 compare with iPad Pro?"
q2 = "Why did Microsoft choose core m3 and not core i3 home Surface Pro 4?"


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


pred = model.predict([sample_1, sample_2], verbose = 1)
pred += model.predict([sample_2, sample_1], verbose = 1)
pred /= 2
print(pred.ravel())