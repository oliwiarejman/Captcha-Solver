import os
import numpy as np
from PIL import Image
from keras.src.utils import img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(data_path, image_size=(30, 50), test_size=0.2, random_state=42):
    X = []
    y = []

    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            img_path = os.path.join(data_path, filename)
            image = Image.open(img_path).convert('L')
            x = [img_to_array(image.crop((image_size[0]*i, 0, image_size[0]*(i+1), image_size[1]))) for i in range(len(filename[:-4]))]
            X.extend(x)
            y.extend([c for c in filename[:-4]])

    X = np.array(X)
    y = np.array(y)

    X = X.astype('float32') / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder

data_path = "samples"
X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data_path)
