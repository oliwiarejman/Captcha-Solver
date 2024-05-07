import os
import numpy as np
from PIL import Image
import keras
from keras.src.models import load_model
from keras.src.utils import img_to_array
from preprocess import preprocess_data
from train import train_model

def solve_captcha(img_path, model, label_encoder):
    image = Image.open(img_path).convert('L')
    x = [img_to_array(image.crop((30*i, 0, 30*(i+1), 50))) for i in range(5)]
    X_pred = np.array(x)
    X_pred = X_pred.astype('float32') / 255.0
    y_pred = model.predict(X_pred)
    predicted_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
    captcha_text = ''.join(predicted_labels)
    print('Predicted captcha:', captcha_text)

if __name__ == "__main__":
    data_path = "samples"
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data_path)
    trained_model, training_history = train_model(X_train, X_test, y_train, y_test)

    img_path = 'samples/2b827.png'
    solve_captcha(img_path, trained_model, label_encoder)
