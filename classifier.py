from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import os, joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
# vectorizer
vectorizer = open(os.path.join("static/models/count_vectoriser.pkl"),"rb")
gen_cv = joblib.load(vectorizer)

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():


    name = request.form['namequery'].capitalize() # Name is from the the form of  the index html
    modelchoice = request.form['model']

    if modelchoice == 'nb':
        gen_nb_model = open(os.path.join("static/models/nb_model_predict.pkl"), "rb")
        gen_clf = joblib.load(gen_nb_model)
    elif modelchoice == 'logit':
        gen_logit_model = open(os.path.join("static/models/log_model_predict.pkl"), "rb")
        gen_clf = joblib.load(gen_logit_model)


    # Predictig the value
    data = [name]
    vect = gen_cv.transform(data).toarray()
    my_prediction = gen_clf.predict(vect)
    return render_template('results.html',prediction = my_prediction,name=name)



if __name__ == '__main__':
    app.run(debug=True)