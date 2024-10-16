import pickle
from flask import Flask
from flask import request
from flask import jsonify
#
with open("dt_model.bin", 'rb') as f_in:
    dv, pipeline = pickle.load(f_in)

app = Flask('Houses')


@app.route('/predict', methods=['POST'])
def predict():
    session = request.get_json()

    # Transform the input data
    X = dv.transform([session])

    # Predict the price
    y_pred = pipeline.predict(X)[0]

    result = {
        'predicted_price': float(y_pred)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
