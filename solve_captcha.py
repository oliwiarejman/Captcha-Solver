import os
import numpy as np
from PIL import Image
# from keras.src.models import load_model
from preprocess import preprocess_data
from train import train_model, trained_model

def solve_captchas(data_path, model, label_encoder, output_file):
    filenames = os.listdir(data_path)
    with open(output_file, 'w') as f:
        for filename in filenames:
            img_path = os.path.join(data_path, filename)
            image = Image.open(img_path).convert('L')
            x = [np.array(image.crop((30*i, 0, 30*(i+1), 50))) for i in range(5)]
            X_pred = np.array(x)
            X_pred = X_pred.astype('float32') / 255.0
            y_pred = model.predict(X_pred)
            predicted_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
            captcha_text = ''.join(predicted_labels)
            f.write(f"{filename[:-4]}: {captcha_text}\n")

if __name__ == "__main__":
    data_path = "samples"
    output_file = "captcha_results.txt"
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data_path)
    trained_model, training_history = train_model(X_train, X_test, y_train, y_test)

    solve_captchas(data_path, trained_model, label_encoder, output_file)