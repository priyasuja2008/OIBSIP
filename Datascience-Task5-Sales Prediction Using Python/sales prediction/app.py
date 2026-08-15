from flask import Flask, render_template, request
import joblib
import pandas as pd

# Create Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load("sales_prediction_model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        tv = float(request.form["tv"])
        radio = float(request.form["radio"])
        newspaper = float(request.form["newspaper"])

        # Create DataFrame with the same column names used during training
        input_data = pd.DataFrame({
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        })

        # Predict sales
        prediction = model.predict(input_data)[0]

        # Round prediction to 2 decimal places
        prediction = round(prediction, 2)

        # Render the result on the same page
        return render_template(
            "index.html",
            prediction=prediction,
            tv=tv,
            radio=radio,
            newspaper=newspaper
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Error: {str(e)}"
        )


# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)