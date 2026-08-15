from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    sepal_length = float(request.form["sl"])
    sepal_width = float(request.form["sw"])
    petal_length = float(request.form["pl"])
    petal_width = float(request.form["pw"])

    prediction = model.predict([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    # Change these if your model uses different labels
    classes = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica"
    ]

    result = classes[int(prediction[0])]

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)