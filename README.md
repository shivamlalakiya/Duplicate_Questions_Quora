## Quora Question Pairs - Predicting Duplicate Questions
<p align="justify">
This project aims to predict whether a pair of questions from the Quora Question Pairs dataset have the same meaning. We use a combination of Natural Language Processing (NLP) techniques and machine learning algorithms to preprocess and model the data.
</p>

## Introduction
<p align="justify">
The Quora Question Pairs dataset contains a set of question pairs with binary labels indicating whether they have the same meaning or not. The ground truth labels are noisy, subjective, and inherently difficult to determine, and therefore, our goal is to train a model that can generalize well on unseen data.
</p>

<p align="justify">
We explore two NLP techniques, Word2Vec and LSTM, to extract features from the text data and feed them into a binary classifier. Word2Vec is a word embedding algorithm that learns vector representations of words based on their context. At the same time, LSTM is a recurrent neural network architecture that can model sequences and dependencies in text data.
</p>

## Data
<p align="justify">
The dataset consists of two files:
train.csv - the training set contains a set of question pairs and their corresponding labels indicating whether the questions have the same meaning.  
</p>
**The link to download the file:**   
https://www.kaggle.com/competitions/quora-question-pairs/data?select=train.csv.zip 

**Link to download Google News vector negative 300:** https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

## Preprocessing
<p align="justify">
We first preprocess the text data by removing stop words and punctuations and converting the text to lowercase. We then tokenize the text data and create sequences of fixed length. For each word in the sequences, we use Word2Vec to obtain a vector representation, which captures its context within the text corpus.
</p>
## Model
<p align="justify">
We explore two models, one based on Word2Vec and one based on LSTM, to predict whether two questions have the same meaning. The Word2Vec-based model concatenates the vector representations of the two questions in the pair and feeds them into a binary classifier, which outputs a probability of having the same meaning.
</p>
<p align="justify">
The LSTM-based model uses a recurrent neural network architecture to model the sequence of words in the question pair. The input sequence is first embedded using an embedding layer, then multiple LSTM layers, and finally, a binary classification layer.
</p>
## Evaluation
<p align="justify">
We evaluate the models using log loss, the evaluation metric used in the Kaggle competition. We train the models on the training set and tune the hyperparameters using cross-validation. Finally, we predict the labels for the test set and submit them to the Kaggle competition for evaluation.
</p>
## Conclusion
<p align="justify">
In this project, we explored two NLP techniques, Word2Vec and LSTM, to predict whether two questions have the same meaning. We preprocessed the text data by removing stop words and punctuations and creating fixed-length sequences. We trained two models, one based on Word2Vec and one based on LSTM, to predict the labels. The models were evaluated using log loss, and the predictions were submitted to the Kaggle competition for evaluation.
</p>
<p align="justify">
Our results suggest that the LSTM-based model outperforms the Word2Vec-based model, achieving a log loss of X on the test set. This indicates that modeling the sequence of words in the question pairs can capture important information about their meaning and is a promising direction for future research.
</p>
## Example
