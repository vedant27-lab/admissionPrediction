from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and encoder
model = joblib.load("admission_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get input values from the form
            year = int(request.form["year"])
            tenth_marks = float(request.form["tenth_marks"])
            twelfth_marks = float(request.form["twelfth_marks"])
            twelfth_div = int(request.form["twelfth_div"])
            aieee_rank = int(request.form["aieee_rank"])

            # Make prediction
            user_input = np.array([[year, tenth_marks, twelfth_marks, twelfth_div, aieee_rank]])
            pred_index = model.predict(user_input)
            predicted_college = le.inverse_transform(pred_index)[0]

            return render_template("index.html", prediction=predicted_college)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
