from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load your Keras model here
model = tf.keras.models.load_model('best_model.h5')







if __name__ == '__main__':
    app.run(debug=True)
