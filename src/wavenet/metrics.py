import keras.backend as K

def categorical_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1)))