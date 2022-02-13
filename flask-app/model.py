import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix

from config import SPAM, NOT_SPAM, LOGISTIC_REGRESSION, SUPPORT_VECTOR_MACHINE, DEFAULT_MODEL_CONFIG

def lowercase(data):
    data = data.lower()
    return data

class ClassificationModel:
    def __init__(self, config = None):
        # read data
        data = pd.read_csv('../SMSSpamCollection.txt', sep='\t', names=['category', 'SMS'])
        # preprocess
        data['SMS'] = data['SMS'].apply(lowercase)
        # normalise
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(data['SMS'])
        # model configuration
        if config is None:
            config = DEFAULT_MODEL_CONFIG
        # fit model
        if config['classifier'] == LOGISTIC_REGRESSION:
            # logistic regression classifier with regularization to avoid overfitting
            logit = LogisticRegression(C=config['regularization'])
            logit.fit(X, data['category'])
            self.model = logit
        elif config['classifier'] == SUPPORT_VECTOR_MACHINE:
            # SVM classifier with regularization, probabilities enabled for obtaining confidence score
            svm_classifier = svm.SVC(C=config['regularization'], probability=True)
            svm_classifier.fit(X, data['category'])
            self.model = svm_classifier
        
    def classify(self, input_string):
        input_string = self.vectorizer.transform([input_string])
        spam_score = self.model.predict_proba(input_string)[0][1]
        if spam_score > 0.7:
            return SPAM, spam_score
        else:
            return NOT_SPAM, spam_score

