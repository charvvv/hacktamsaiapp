from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS

# Load models
model1 = tf.keras.models.load_model('history_Stroke.h5')
model2 = tf.keras.models.load_model('history_Cardio.h5')
model3 = tf.keras.models.load_model('history_CFC.h5')

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Health Diagnosis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array([list(data.values())])

        pred1 = model1.predict(input_data).tolist()
        pred2 = model2.predict(input_data).tolist()
        pred3 = model3.predict(input_data).tolist()

        return jsonify({
            "model1_prediction": pred1,
            "model2_prediction": pred2,
            "model3_prediction": pred3
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
