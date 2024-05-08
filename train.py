from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.src.optimizers import Adam
from preprocess import preprocess_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(50, 30, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1500))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(36))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=200, epochs=300, validation_data=(X_test, y_test), shuffle=True)

    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.save('captcha_model.h5')

    return model, history

def plot_training_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

if __name__ == "__main__":
    data_path = "samples"
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data_path)
    trained_model, training_history = train_model(X_train, X_test, y_train, y_test)

    plot_training_history(training_history)

    y_pred = trained_model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    print("Predictions:", y_pred_labels)

    # Wizualizacja macierzy błędu
    labels = [str(label) for label in label_encoder.classes_]
    plot_confusion_matrix(y_test, y_pred_labels, labels)