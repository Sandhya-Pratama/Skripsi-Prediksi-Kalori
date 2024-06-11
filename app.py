from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib
from flask_cors import CORS

# Load the model
model = pickle.load(open('model/pca+svr+gridsearchcv.pkl', 'rb'))


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

@app.route('/')
def index():
    dataset = pd.read_csv('calories+exercise.csv')
    data = dataset.to_dict('records')
    return render_template('index.html', data=data)  

@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = (
            float(request.form['User_ID']),
            float(request.form['Gender']),
            float(request.form['Age']),
            float(request.form['Height']),
            float(request.form['Weight']),
            float(request.form['Duration']),
            float(request.form['Heart_Rate']),
            float(request.form['Body_Temperature']),
        )
        # Convert data to numpy array
        data_array = np.asarray(data)
        
        # Select only the necessary features (adjust the indices based on your model)
        selected_data = data_array[[0, 1, 2, 3, 4, 5, 6, 7]]  # Selecting all features
        
        # Reshape data for prediction
        data_reshape = selected_data.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(data_reshape)
        
        # Return the result
        result = prediction[0]  # For regression, we return the predicted value directly
        return jsonify(result=result)
    except Exception as e:
        return jsonify(result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
