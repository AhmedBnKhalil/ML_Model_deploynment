import pickle

import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# load pickle model
model = pickle.load(open('heart_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 1:
        prediction ='POSITIVE'
    elif prediction == 0:
        prediction = 'NEGATIVE'

    return render_template('index.html', prediction_text="This person prediction is {}".format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
