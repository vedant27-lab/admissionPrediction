from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('admission_model.pkl', 'rb'))

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

            # Prepare input for model
            features = np.array([[year, tenth_marks, twelfth_marks, twelfth_div, aieee_rank]])

            # Make prediction
            predicted_college = model.predict(features)[0]

            prediction = f"{predicted_college}"  # Convert to string
        except Exception as e:
            error = f"Invalid input: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
