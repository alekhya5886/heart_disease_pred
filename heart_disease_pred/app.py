from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

# Define the home route
@app.route('/')
def index():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    features = {
        'Gender': int(request.form['gender']),
        'Age': int(request.form['age']),
        'FamilyHistory': int(request.form['familyHistory']),
        'PastHeartIssues': int(request.form['pastHeartIssues']),
        'PhysicalActivity': int(request.form['physicalActivity']),
        'Diet': int(request.form['diet']),
        'Smoking': int(request.form['smoking']),
        'AlcoholConsumption': int(request.form['alcoholConsumption']),
        'SleepPatterns': int(request.form['sleepPatterns']),
        'HighBloodPressure': int(request.form['highBloodPressure']),
        'Diabetic': int(request.form['diabetic']), 
    }

    # Create a DataFrame from user input
    input_data = pd.DataFrame([features])

    # Make a prediction using the trained model
    prediction = model.predict(input_data)[0]

    # Display the prediction on the webpage
    return render_template('index.html', prediction=f"The user is {'prone' if prediction else 'not prone'} to heart disease.")

if __name__ == '__main__':
    app.run(debug=True)
