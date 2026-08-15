from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("car_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from the form
        car_name = request.form["car_name"]
        brand = request.form["brand"]
        model_name = request.form["model"]
        min_cost_price = float(request.form["min_cost_price"])
        max_cost_price = float(request.form["max_cost_price"])
        vehicle_age = float(request.form["vehicle_age"])
        km_driven = float(request.form["km_driven"])
        seller_type = request.form["seller_type"]
        fuel_type = request.form["fuel_type"]
        transmission_type = request.form["transmission_type"]
        mileage = float(request.form["mileage"])
        engine = float(request.form["engine"])
        max_power = float(request.form["max_power"])
        seats = float(request.form["seats"])

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "car_name": car_name,
            "brand": brand,
            "model": model_name,
            "min_cost_price": min_cost_price,
            "max_cost_price": max_cost_price,
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats
        }])

        # Predict price
        prediction = model.predict(input_data)[0]

        # Format prediction
        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)