# def train(classifier_type = "logistic"):
#     sms_data = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=['category', 'SMS'])
#     sms_data['SMS'] = sms_data['SMS'].apply(preprocess)
#     X_train, X_test, y_train, y_test = train_test_split(sms_data['SMS'], sms_data['category'], test_size = 0.1, random_state = 1)
#     vectorizer = TfidfVectorizer()
#     X_train = vectorizer.fit_transform(X_train)
#     if classifier_type == "logistic": 
#         # logistic regression classifier with regularization to avoid overfitting
#         logit = LogisticRegression(C=5e1)
#         logit.fit(X_train, y_train)
#         X_test = vectorizer.transform(X_test)
#         class_probabilities = logit.predict_proba(X_test)
#         print(len(class_probabilities))
#         spam_scores = [ value[1] for value in class_probabilities]
#         # X_test = pd.DataFrame(X_test).reset_index(drop=True)
#         # X_test["spam_score"] = spam_scores
#         y_pred1 = logit.predict(X_test)
#         y_pred2 = ["spam" if score > 0.7 else "ham" for score in spam_scores]
#         print(confusion_matrix(y_test, y_pred1))
#         print(confusion_matrix(y_test, y_pred2))
#         return logit, vectorizer
#     elif classifier_type == "svm":
#         svm_model = svm.SVC(C=1000, probability=True)
#         svm_model.fit(X_train, y_train)
#         X_test = vectorizer.transform(X_test)
#         y_pred = svm_model.predict(X_test)
#         class_probabilities = svm_model.predict_proba(X_test)
#         for p in class_probabilities:
#             if p[0] > 0.1 and p[0] < 0.9:
#                 print(p)
#         print(class_probabilities)
#         print(confusion_matrix(y_test, y_pred))
#         return svm_model, vectorizer

# def classify(input_string):
#     return NOT_SPAM

# print("Hello, please enter an SMS message:")
# msg = input()
# logit, vectorizer = train("logistic")
# msg = vectorizer.transform([msg])
# print(msg)
# probs = logit.predict_proba(msg)
# prediction = logit.predict(msg)
# print(probs)
# if probs[0][1] > 0.7:
#     print("SPAM")
# else:
#     print("NOT SPAM")
# print(prediction)