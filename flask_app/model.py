""" ClassificationModel Class and utilities for building the model """
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from config import SPAM, NOT_SPAM, LOGISTIC_REGRESSION, SUPPORT_VECTOR_MACHINE, DEFAULT_MODEL_CONFIG

def lowercase(data):
    """ Convert given input string to lowercase """
    data = data.lower()
    return data

class ClassificationModel:
    """ Class that decribes the spam classification model """
    def __init__(self, config = None):
        """ Initialise model attributes and fit model """
        # model configuration
        if config is None:
            config = DEFAULT_MODEL_CONFIG
        # read data
        data = pd.read_csv('../SMSSpamCollection.txt', sep='\t', names=['category', 'SMS'])
        # preprocess
        data['SMS'] = data['SMS'].apply(lowercase)
        # normalise
        if config['remove_stopwords']:
            self.vectorizer = TfidfVectorizer(stop_words='english')
        else:
            self.vectorizer = TfidfVectorizer()
        vectorized_data = self.vectorizer.fit_transform(data['SMS'])
        # fit model
        if config['classifier'] == LOGISTIC_REGRESSION:
            # logistic regression classifier with regularization to avoid overfitting
            logit = LogisticRegression(C=config['regularization'])
            logit.fit(vectorized_data, data['category'])
            self.model = logit
        elif config['classifier'] == SUPPORT_VECTOR_MACHINE:
            # SVM classifier with regularization and
            # probabilities enabled for obtaining confidence score
            svm_classifier = svm.SVC(C=config['regularization'], probability=True)
            svm_classifier.fit(vectorized_data, data['category'])
            self.model = svm_classifier

    def classify(self, input_string):
        """ Classify input string as SPAM or NOT SPAM """
        input_string = self.vectorizer.transform([input_string])
        spam_score = self.model.predict_proba(input_string)[0][1]
        if spam_score > 0.7:
            return SPAM, spam_score
        return NOT_SPAM, spam_score
