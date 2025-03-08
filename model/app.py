from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load('admission_model.pkl')
le = joblib.load('label_encoder.pkl')

def is_valid_input(year, tenth_marks, twelfth_marks, twelfth_div, aieee_rank):
    return (year > 0 and 0 <= tenth_marks <= 100 and 0 <= twelfth_marks <= 100 and 
            1 <= twelfth_div <= 3 and aieee_rank > 0)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Get user input
            year = float(request.form['year'])
            tenth_marks = float(request.form['tenth_marks'])
            twelfth_marks = float(request.form['twelfth_marks'])
            twelfth_div = float(request.form['twelfth_div'])
            aieee_rank = float(request.form['aieee_rank'])

            if is_valid_input(year, tenth_marks, twelfth_marks, twelfth_div, aieee_rank):
                # Prepare input for model
                features = np.array([[year, tenth_marks, twelfth_marks, twelfth_div, aieee_rank]])

                # Make prediction
                predicted_college_index = model.predict(features)
                predicted_college = le.inverse_transform(predicted_college_index)

                prediction = f"{predicted_college[0]}"  # Convert to string
            else:
                error = "Invalid input: Please make sure all inputs are within the valid range."
        except Exception as e:
            error = f"Invalid input: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
