import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import nltk
from collections import Counter

def read_file(path):

    df = pd.read_csv(path, encoding = "ISO-8859-1",on_bad_lines='skip')
    df = df.fillna(method='ffill')
    # df = df[:30000]

    return df

def collate(dataframe):
    agg_func = lambda s: [(w, pos, t, ) for w, pos, t in zip(s['Word'].values.tolist(), s['POS'].values.tolist(), s['Tag'].values.tolist())]
    grouped = dataframe.groupby('Sentence #').apply(agg_func)
    return list(grouped)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'word.lower()': word.lower(),  # the word in lowercase
        'word[-3:]': word[-3:],  # last three characters
        'word[-2:]': word[-2:],  # last two characters
        'word.isupper()': word.isupper(),  # true, if the word is in uppercase
        'word.istitle()': word.istitle(),  # true, if the first character is in uppercase and remaining characters are in lowercase
        'word.isdigit()': word.isdigit(),  # true, if all characters are digits
        'postag': postag,  # POS tag
        'postag[:2]': postag[:2],  # IOB prefix
    }
    
    if i > 0:
        word1 = sent[i-1][0]  # the previous word
        postag1 = sent[i-1][1]  # POS tag of the previous word
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })  # add some features of the previous word

        if i > 1:
            word1 = sent[i-2][0]  # the previous word
            postag1 = sent[i-2][1]  # POS tag of the previous word
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })  # add some features of the previous word

        else: features['BOS_2'] = True

    else:
        features['BOS'] = True  # BOS: begining of the sentence
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]  # the next word
        postag1 = sent[i+1][1]  # POS tag of the next next word
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })  # add some features of the next next word

        if i < len(sent)-2:
            word1 = sent[i+2][0]  # the next word
            postag1 = sent[i+2][1]  # POS tag of the next next word
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })  # add some features of the next next word

        else: features['EOS_2'] = True

    else:
        features['EOS'] = True  # EOS: end of the sentence

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for _, _, label in sent]

def crf_model(X_train,y_train):

    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',  # l2sgd: Stochastic Gradient Descent with L2 regularization term
        max_iterations=1000,  # maximum number of iterations
        )
    try:
        crf.fit(X_train, y_train)
    except AttributeError:
        pass
    return crf

def crf_train(filepath):

    df = read_file(filepath)
    sentences = collate(df)

    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=1)

    # X and y in the required format
    X_train, y_train = [sent2features(s) for s in train_sentences], [sent2labels(s) for s in train_sentences]
    # X_test, y_test = [sent2features(s) for s in test_sentences], [sent2labels(s) for s in test_sentences]

    crf = crf_model(X_train,y_train)

    pickle.dump(crf, open('crf_model.sav', 'wb')) 


def crf_pred(test_sentence = "Who is the director of Star Wars: Episode VI - Return of the Jedi?"):
        
    loaded_model = pickle.load(open('crf_model.sav', 'rb'))

    test_tokens = nltk.word_tokenize(test_sentence)
    test = sent2features(nltk.pos_tag(test_tokens)) 
    test = [test]

    y_pred = loaded_model.predict(test)
    print(y_pred)


if __name__ == "__main__":

    text = ['Who is the director of Good Will Hunting?',
        'Who directed The Bridge on the River Kwai?',
        "Who is the director of Star Wars: Episode VI - Return of the Jedi?",
        "What is the genre of Good Neighbors?",
        "Who is the screenwriter of The Masked Gang: Cyprus?",
        "What is the MPAA film rating of Weathering with You?",
        "Show me a picture of Halle Berry.",
        "What does Julia Roberts look like?",
        "Let me know what Sandra Bullock looks like.",
        'Recommend movies similar to Hamlet and Othello.',
        'Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?',
        'Recommend movies like Nightmare on Elm Street, Friday the 13th, and Halloween.',
        'What is the box office of The Princess and the Frog?',
        'Can you tell me the publication date of Tom Meets Zizou?',
        'Who is the executive producer of X-Men: First Class?']

    for te in text:
        print(te)
        print(crf_pred(te),'\n')
