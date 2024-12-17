from flask import Flask, request, jsonify, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('svm_spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input email text from the form
    input_text = request.form['email_text']
    
    # Transform input using the vectorizer
    input_vectorized = vectorizer.transform([input_text])
    
    # Predict using the loaded model
    prediction = model.predict(input_vectorized)
    
    # Return result
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
