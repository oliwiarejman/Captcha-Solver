from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.src.optimizers import Adam
from preprocess import preprocess_data

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
    model.add(Dense(36))  # 36 klas: 26 liter + 10 cyfr
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=150, epochs=200, validation_data=(X_test, y_test), shuffle=True)

    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.save('captcha_model.h5')

    return model, history


if __name__ == "__main__":
    data_path = "samples"
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data_path)
    trained_model, training_history = train_model(X_train, X_test, y_train, y_test)
