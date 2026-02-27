from flask import Flask, render_template, request
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("floods.save")
scaler = joblib.load("transform.save")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            values = [float(x) for x in request.form.values()]
            data = np.array(values).reshape(1, -1)

            # Scale input
            data = scaler.transform(data)

            # Predict class
            prediction = model.predict(data)

            # Predict probability
            probability = model.predict_proba(data)[0][1] * 100

            if prediction[0] == 1:
                return render_template("chance.html", confidence=round(probability, 2))
            else:
                return render_template("noChance.html", confidence=round(100 - probability, 2))

        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
