# pylint: skip-file

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize 
nltk.download('punkt')
from config import SUPPORT_VECTOR_MACHINE, LOGISTIC_REGRESSION

def lowercase(data):
    """ Convert given input string to lowercase """
    data = data.lower()
    return data

def validate_model(data, config):
    # preprocess
    data['SMS'] = data['SMS'].apply(lowercase)
    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(data['SMS'], data['category'], test_size = 0.1, random_state = 1)
    # normalise
    if config['remove_stopwords']:
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    # fit model on training data
    model = None
    if config['classifier'] == LOGISTIC_REGRESSION:
        # logistic regression classifier with regularization to avoid overfitting
        logit = LogisticRegression(C=config['regularization'], solver='liblinear')
        logit.fit(X_train, y_train)
        model = logit
    elif config['classifier'] == SUPPORT_VECTOR_MACHINE:
        # SVM classifier with regularization and
        # probabilities enabled for obtaining confidence score
        svm_classifier = svm.SVC(C=config['regularization'], probability=True)
        svm_classifier.fit(X_train, y_train)
        model = svm_classifier
    if model is not None:
        X_test = vectorizer.transform(X_test)
        # predicting directly using predict()
        y_pred1 = model.predict(X_test)
        # spam if spam score > 0.7, else ham
        class_probabilities = model.predict_proba(X_test)
        spam_scores = [value[1] for value in class_probabilities]
        y_pred2 = ["spam" if score > 0.7 else "ham" for score in spam_scores]
        # printing results
        # print("Confusion Matrix: Using predict()")
        # print(confusion_matrix(y_test, y_pred1))
        print("Confusion Matrix: Predicting SPAM for spam score > 0.7")
        cm = confusion_matrix(y_test, y_pred2)
        print(cm)
        sns.heatmap(cm/np.sum(cm), cmap='Blues', fmt='.2%', annot=True)
        plt.show()

# read data
data = pd.read_csv('../SMSSpamCollection.txt', sep='\t', names=['category', 'SMS'])
print(data.head())
print("Number of SMS messages: {len}".format(len=len(data)))

# categorical plot for spam and ham counts
bars = sns.catplot(x="category", kind="count", data=data)
plt.title("Number of Spam and Ham SMS Messages")
plt.show()

# get average number of words per SMS message
messages = list(data['SMS'])
messages_in_words = [msg.split(' ') for msg in messages]
avg = sum(len(msg) for msg in messages_in_words)/len(messages_in_words)
print("Average number of words per SMS message: {avg}".format(avg=avg))

# get number of stopwords
stopwords_per_message = []
for msg in messages:
    word_tokens = word_tokenize(msg)
    stopwords_msg = [w for w in word_tokens if w in stopwords.words('english')]
    stopwords_per_message.append(len(stopwords_msg))

print(stopwords_per_message)
print(sum(stopwords_per_message)/len(stopwords_per_message))

# get strings of words from spam and ham messages
spam_data = data[data['category'] == 'spam']
spam_words = " ".join(list(spam_data['SMS']))
ham_data = data[data['category'] == 'ham']
ham_words = " ".join(list(ham_data['SMS']))

# generate word clouds
# take relative word frequencies into account, lower max_font_size
# ham
wordcloud_ham = WordCloud(background_color="white",max_words=len(ham_words),max_font_size=40, relative_scaling=.5).generate(ham_words)
plt.figure()
plt.imshow(wordcloud_ham)
plt.axis("off")
plt.show()

# spam
wordcloud_spam = WordCloud(background_color="white",max_words=len(spam_words),max_font_size=40, relative_scaling=.5).generate(spam_words)
plt.figure()
plt.imshow(wordcloud_spam)
plt.axis("off")
plt.show()

# model exploration and tuning parameters
print("Logistic Regression with stopwords removed")
validate_model(data, {
    'remove_stopwords': True,
    "classifier": LOGISTIC_REGRESSION,
    "regularization": 5e1
})

print("Logistic Regression without stopwords removed")
validate_model(data, {
    'remove_stopwords': False,
    "classifier": LOGISTIC_REGRESSION,
    "regularization": 5e1
})

print("SVM with stopwords removed")
validate_model(data, {
    'remove_stopwords': True,
    "classifier": SUPPORT_VECTOR_MACHINE,
    "regularization": 5e1
})

print("SVM without stopwords removed")
validate_model(data, {
    'remove_stopwords': False,
    "classifier": SUPPORT_VECTOR_MACHINE,
    "regularization": 5e1
})
