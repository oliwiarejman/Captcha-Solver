import os
import numpy as np
from PIL import Image
from keras.src.utils import img_to_array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(data_path):
    X = []
    y = []

    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            img_path = os.path.join(data_path, filename)
            image = Image.open(img_path).convert('L')
            x = [img_to_array(image.crop((30*i, 0, 30*(i+1), 50))) for i in range(5)]
            X.extend(x)
            y.extend([c for c in filename[:-4]])

    X = np.array(X)
    y = np.array(y)

    X = X.astype('float32') / 255.0

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder


