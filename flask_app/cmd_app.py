""" Command Line Application """
from model import ClassificationModel

# build model based on configuration
model = ClassificationModel()

# obtain input text
print("Enter input text:")
input_text = input()

# classify
result, spam_score = model.classify(input_text)

# display result
print("Classification: {result}".format(result=result))
print("Spam Score: {:.2f}".format(spam_score))
