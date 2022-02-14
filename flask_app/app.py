"""Flask App"""
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from model import ClassificationModel

app = Flask(__name__)
CORS(app)

# builds model based on configuration
model = ClassificationModel()

# Endpoints
@app.route('/')
@cross_origin()
def index():
    """Hello World"""
    return "Hello World!"

@app.route('/api/classify/', methods = ['POST'])
@cross_origin()
def classification_endpoint():
    """ Classifies input string as SPAM or NOT SPAM """
    if request.method == 'POST':
        input_string = request.json['inputstring']
        result, score = model.classify(input_string)
        return jsonify({
            'classification': result,
            'spam_score': score
        })
    return jsonify({})
