""" Configuration and constants """
SPAM = "SPAM"
NOT_SPAM = "NOT SPAM"

LOGISTIC_REGRESSION = "logr"
SUPPORT_VECTOR_MACHINE = "svm"

DEFAULT_MODEL_CONFIG = {
    'remove_stopwords': False,
    "classifier": LOGISTIC_REGRESSION,
    "regularization": 5e1
}
