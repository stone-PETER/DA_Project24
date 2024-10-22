from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Function to load and use the Naive Bayes model
def load_and_predict_naive_bayes(text):
    # Load the saved Naive Bayes model
    with open('naive_bayes_model.pkl', 'rb') as file:
        modelNB = pickle.load(file)

    # Load the vectorizer
    vectorizerNB = CountVectorizer()
    train_dataNB = pd.read_csv('Data_Processing/train_data_processed.csv')
    vectorizerNB.fit(train_dataNB['text'])

    # Predict using the loaded model
    text_vectorized = vectorizerNB.transform([text])
    predictionNB = modelNB.predict(text_vectorized)
    return predictionNB[0]

# Function to load and use the Decision Tree model
def load_and_predict_decision_tree(text):
    # Load the saved Decision Tree model
    with open('id3_model.pkl', 'rb') as file:
        modelD3 = pickle.load(file)

    # Load the vectorizer
    vectorizerD3 = CountVectorizer()
    train_dataD3 = pd.read_csv('Data_Processing/train_data_processed.csv')
    vectorizerD3.fit(train_dataD3['text'])

    # Predict using the loaded model
    text_vectorized = vectorizerD3.transform([text])
    predictionD3 = modelD3.predict(text_vectorized)
    return predictionD3[0]


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['post'])
def predict():
    email = request.form.get('email')
    # predict email
    print(email)
    # prediction= load_and_predict_naive_bayes(email)
    # prediction = 1 if prediction == 1 else -1
    # return render_template('index.html', response=prediction)


if __name__ == "__main__":
    app.run(debug=True)
