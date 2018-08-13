from src.wavenet import wavenet
from src import audio_provider, data_generator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
import tensorflow as tf
from src import settings
from src.utils import plot_utils
import keras

def setup():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.log_device_placement = False
    session = tf.Session(config=config)

    keras.backend.tensorflow_backend.set_session(session)

def train():
    model = wavenet.get()
    model_checkpoint = ModelCheckpoint(
        filepath='models/wavenet_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    callbacks = [model_checkpoint]

    # -----------TEACHING---------------

    train_audio = audio_provider.get_file('audio/train.wav')
    valid_audio = audio_provider.get_file('audio/validate.wav')

    train_X, train_y = data_generator.get(audio=train_audio)
    valid_X, valid_y = data_generator.get(audio=valid_audio)

    history = model.fit(x=train_X,
                        y=train_y,
                        callbacks=callbacks,
                        epochs=settings.epochs,
                        verbose=1,
                        steps_per_epoch=settings.steps_per_epoch,
                        validation_steps=settings.validation_steps,
                        validation_data=(valid_X, valid_y))

    plot_utils.show_history(history)

def predict():
    model = wavenet.load()

    audio = audio_provider.get_file('audio/train.wav')
    X, y = data_generator.get(audio=audio)

    predictions = model.predict(X, batch_size=16)
    predictions = audio_provider.generate(predictions)
    valid_y = audio_provider.generate(y)
    print(valid_y[0,:10])
    print(predictions[0,:10])

if __name__ == "__main__":

    setup()
    predict()








