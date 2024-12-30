from flask import Flask, request, jsonify, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained pipeline
with open('trained_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    try:
        age = float(request.form['age'])
        sex = request.form['sex']
        embarked = request.form['embarked']

        # Create input data for the model
        input_data = [[None, sex, age, None, None, None, embarked]]

        # Predict using the pipeline
        prediction = pipeline.predict(input_data)

        # Return prediction
        result = "Survived" if prediction[0] == 1 else "Did not survive"
        return render_template('index.html', prediction_text=f"Prediction: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
