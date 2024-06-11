from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS

# Load the model
scaler = joblib.load('model/scaler.pkl')
pca_best = joblib.load('model/pca_best.pkl')
svr_model = joblib.load('model/svr_model.pkl')

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract data from JSON
        gender = int(data['Gender'])
        age = int(data['Age'])
        height = int(data['Height'])
        weight = int(data['Weight'])
        duration = int(data['Duration'])
        heart_rate = int(data['Heart_Rate'])
        body_temp = float(data['Body_Temp'])

        # Create a NumPy array from the input data
        input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Apply PCA transformation
        input_data_pca = pca_best.transform(input_data_scaled)

        # Make prediction
        prediction = svr_model.predict(input_data_pca)
        prediction_text = f"Predicted Calories: {prediction[0]:.2f}"

        return jsonify({'prediction_text': prediction_text})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/submit', methods=['POST'])
# def submit():
#     try:
#         # Get form data
#         age = request.form.get('Age')
#         gender = request.form.get('Gender')
#         height = request.form.get('Height')
#         weight = request.form.get('Weight')
#         duration = request.form.get('Duration')
#         heart_rate = request.form.get('Heart_Rate')
#         body_temperature = request.form.get('Body_Temperature')

#         # Convert data to floats and validate ranges
#         try:
#             data = [
#                 float(gender),
#                 float(age),
#                 float(height),
#                 float(weight),
#                 float(duration),
#                 float(heart_rate),
#                 float(body_temperature)
#             ]
#         except ValueError as ve:
#             raise ValueError("Invalid input: all values must be numeric")

#         # Create a numpy array from the data
#         data_array = np.array(data)

#         # Reshape data for prediction
#         data_reshape = data_array.reshape(1, -1)
      
#         # Make prediction
#         prediction = model.predict(data_reshape)

#         # Return the result
#         result = prediction[0]  # For regression, we return the predicted value directly
#         return jsonify(result=result)
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return jsonify(result=f"Error: {e}")