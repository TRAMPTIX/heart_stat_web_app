import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    input_data = [int(x) for x in request.form.values()]


    input_data_as_numpy_array = np.asarray(input_data)


    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        output = "chance of not having arithmia"
    else:
        output= "chance of  having arithmia"

    return render_template('index.html', prediction_text='person has 89.9% {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)