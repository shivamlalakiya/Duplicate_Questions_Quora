import tensorflow as tf
import os
import re
import pickle
from prep import preprocess
import numpy as np
import codecs
import csv
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score,accuracy_score


model = tf.keras.models.load_model('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/LSTM_model', compile=False)

# q1 = input('Enter first question: ')
# q2 = input('Enter second question: ')

# q1 = "How does the Surface Pro himself 4 compare with iPad Pro?"
# q2 = "Why did Microsoft choose core m3 and not core i3 home Surface Pro 4?"
text_1 = []
text_2 = []
labels = []

p = preprocess()
with codecs.open('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/fixed_test.csv', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        text_1.append(p.text_to_worldlist(values[1]))
        text_2.append(p.text_to_worldlist(values[2]))


tokenizer = pickle.load(open('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/tokenizer.pickle','rb'))

s1 = tokenizer.texts_to_sequences(text_1)
s2 = tokenizer.texts_to_sequences(text_2)

data_1 = pad_sequences(s1, maxlen=30)
data_2 = pad_sequences(s2, maxlen=30)


pred = model.predict([data_1, data_2], batch_size = 8192, verbose = 1)
pred += model.predict([data_2, data_2], batch_size = 8192, verbose = 1)
pred /= 2

df = pd.read_csv('/Users/shivamlalakiya/Desktop/Projects/Duplicate_questions_Quora/sample_submission.csv')

final = pd.DataFrame({'label':df['is_duplicate'], 'prediction':pred.ravel()})
predict_test = []
for row in range(len(final)):
    if final['prediction'].iloc[row] > 0.5:
        predict_test.append(1)
    else:
        predict_test.append(0)

acc = accuracy_score(final['label'], predict_test)
f1 = f1_score(final['label'], predict_test)
print(acc, f1)