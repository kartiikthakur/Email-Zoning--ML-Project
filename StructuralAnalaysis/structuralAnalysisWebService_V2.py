#Neural network functions for creating a prediction model
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, LSTM
#Object serialization function
import joblib
import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
#Mathematical functions to handle vectors
import numpy as np
#Operating system functions for interacting with folders and files
import codecs
import errno
import os
import sys
import traceback
from keras.models import model_from_json
from flask import Flask, jsonify, request, abort, make_response
from flask_basicauth import BasicAuth
from keras import backend as K
import json
import re



#Create a flask server
app = Flask(__name__)

app.config['BASIC_AUTH_USERNAME'] = 'panacea'
app.config['BASIC_AUTH_PASSWORD'] = 'classify'

basic_auth = BasicAuth(app)

path = os.getcwd()

global wordEmbeddingModel
with open(path + '/StructuralAnalaysis_V2/wordEmbeddingModelSize-40',"rb") as file:
    wordEmbeddingModel = joblib.load(file)

global structuralLabels
structuralLabels = ["Body", "Body/Intro", "Body/Outro"]

global structureMapping
structureMapping = {0:"Body", 1:"Body/Intro", 2:"Body/Outro"}

global emailLength
#Best average email length (matrix rows)
emailLength = 20

global featureNumber
#Best number of features (marix columns)
featureNumber = 40


#Loads a classification model from a file
#Input: fileName - string
#Ouput: class object
def loadModel(fileName):
    jsonFile = open(fileName + ".json", 'r')
    jsonModel = jsonFile.read()
    jsonFile.close()
    classifier = model_from_json(jsonModel)
    # load weights into new model
    classifier.load_weights(fileName + '.h5')
    # Compiling the neural network
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier


def textPredict(model,matrix, position, lineLength, bodyLength, text, Body, Body_Intro, Body_Outro):
    print(text)
    probability = model.predict(np.array([matrix]))
    prediction = structureMapping[np.argmax(probability)]
    print(prediction)
    introKeyword = ['hi', 'dear', 'to', 'hey', 'hello', 'thanks', 'good morning', 'good afternoon', 'good evening']
    outroKeyword = ['warm regards', 'kind regards', 'regards', 'cheers', 'many thanks', 'thanks', 'thank',
                    'sincerely', 'ciao', 'best regards', 'bgif', 'thank you', 'thankyou', 'talk soon', 'cordially',
                    'yours truly', 'thanking you', 'sent from', 'all the best']
    if prediction == 'Body':
        flag = 0
        if len(text.split()) <= 4 and bodyLength > 5:
            if position < 2 and len(Body_Outro) == 0:
                prediction = 'Body/Intro'
            elif position > bodyLength - 1 and len(Body) != 0:
                prediction = 'Body/Outro'
            else:
                for word in introKeyword:
                    if word in text.split():
                        flag = 1
                        break
                for word in outroKeyword:
                    if word in text.split():
                        flag = 2
                        break
                if flag == 1:
                    prediction = 'Body/Intro'
                elif flag == 2:
                    prediction = 'Body/Outro'
    elif prediction == 'Body/Outro':
        if (lineLength < 10 or len(text.split()) <= 4) and position <= 2:
            prediction = 'Body/Intro'
        elif (len(text.split()) >= 8):
            flag = 1
            for word in outroKeyword:
                if word in text.split():
                    flag = 0
                    break
            if flag == 1:
                prediction = 'Body'
        if prediction == 'Body/Outro':
            if bodyLength >= 30 and position <= int(bodyLength * 0.75):
                prediction = 'Body'
            if len(Body) == 0 and len(Body_Intro) == 0:
                if len(text.split()) <= 4:
                    prediction = 'Body/Intro'
                else:
                    prediction = 'Body'
    elif prediction == 'Body/Intro':
        if (lineLength < 10 or len(text.split()) <= 4) and position > bodyLength - 1:
            prediction = 'Body/Outro'
        elif (len(text.split()) >= 8) and position > 1:
            flag = 1
            for word in introKeyword:
                if word in text.split():
                    flag = 0
                    break
            if flag == 1:
                prediction = 'Body'
    return prediction


def createMatrix(text):
    # Create test matrix
    text = text.split(" ")
    matrix = []
    temp = 0
    for word in text:
        if temp < emailLength:
            try:
                matrix.append(wordEmbeddingModel[word])
                temp += 1
            except:
                pass
        else:
            break
    missingElements = emailLength - len(matrix)
    for _ in range(missingElements):
        # Create vectors of 0's of the length of our word embeddings for missing elements
        matrix.append([0.0 for _ in range(featureNumber)])
    return matrix


@app.route('/PANACEA/structuralAnalysis', methods=['POST'])
@basic_auth.required
def initialEmailAnalysisRawEmail():
    modelName = path + '/StructuralAnalaysis_V2/RNNWordEmbeddingModel'
    K.clear_session()
    Body = []
    Body_Intro = []
    Body_Outro = []
    model = loadModel(modelName)
    lastPrediction = ''
    # Get raw email from system (PANACEA)
    try:
        content = request.data
        print("The content has arrived")
        content = content.decode('utf-8')
        # print(content)
    except Exception as e:
        print(e)
        abort(400, description="Unable to fetch email")

    content = content.replace('\\n', '\n')
    lines = content.split('\n')

    # print(content)
    # print(lines)
    recordNum = 0
    while recordNum < len(lines):
        if len(lines[recordNum].split()) == 0:
            lines.pop(recordNum)
            recordNum -= 1
        else:
            break
        recordNum += 1

    for i in range(len(lines) - 1, -1, -1):
        if len(lines[i].split()) == 0:
            lines.pop(i)
        else:
            break

    lineNum = 1
    for line in lines:
        text = line.lower()
        text = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            "", text)
        text = text.replace('\\r', '').replace('\\t', '')
        text = re.sub(r"\S*@\S*\s?", "", text)
        text = re.sub(r"[^A-Za-z0-9\n]+", " ", text)

        # print(line.split())
        # print(len(line.split()))
        if len(text.split()) != 0:
            matrix = createMatrix(text)
            try:
                prediction = textPredict(model, matrix, lineNum, len(text), len(lines), text, Body, Body_Intro, Body_Outro)
                if lastPrediction != '':
                    if lastPrediction == 'Body/Intro' and prediction == 'Body/Outro':
                        if lineNum > int(len(lines) * 0.75):
                            prediction = 'Body'
                        else:
                            prediction = 'Body/Intro'
                    elif lastPrediction == 'Body/Outro' and prediction == 'Body/Intro':
                        prediction = 'Body/Outro'
                    elif lastPrediction == 'Body/Outro' and prediction == 'Body' and lineNum > int(len(lines) * 0.8):
                        prediction = 'Body/Outro'
                lastPrediction = prediction
            except Exception as e:
                print(e)
                abort(500, description=str(e))
            line = line.replace('\\r', '').replace('\r', '').replace('\\t', '').replace('\t', '').replace('\\', '')
            #print(prediction)
            if prediction == 'Body':
                Body.append(line)
            elif prediction == 'Body/Intro':
                Body_Intro.append(line)
            else:
                Body_Outro.append(line)
        lineNum += 1

    ouputJSON = {'Body/Intro': '\n'.join(Body_Intro), 'Body': '\n'.join(Body), 'Body/Outro': '\n'.join(Body_Outro)}

    return make_response(json.dumps(ouputJSON), 200)


#########################################################
# Flask error handlers
#########################################################

#An error with status code 400 often means bad request.
@app.errorhandler(400)
def errorHandler400(error):
    return make_response(jsonify({'error': error.description}), 400)


#An error with status code 500 often means it's an internal error.
@app.errorhandler(500)
def errorHandler500(error):
    return make_response(jsonify({'error': error.description}), 500)
#########################################################
#
#########################################################


if __name__ == '__main__':
    app.run(debug=True, port=9009, host='0.0.0.0',ssl_context=('cert.pem', 'key.pem'))



