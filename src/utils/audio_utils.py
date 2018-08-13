import scipy.signal
import numpy as np

def ensure_sample_rate(file_sample_rate, dest_sample_rate, audio):
    if file_sample_rate != dest_sample_rate:
        audio = scipy.signal.resample_poly(audio, dest_sample_rate, file_sample_rate)

    return audio


def ensure_mono(audio):
    if audio.ndim == 2:
        audio = audio[:, 0]
    return audio


def float_to_uint8(audio):
    audio += 1.
    audio /= 2.
    uint8_max_value = np.iinfo('uint8').max
    audio *= uint8_max_value
    audio = audio.astype('uint8')
    return audio

def uint8_to_float(audio):
    audio = audio.astype('float')
    uint8_max_value = np.iinfo('uint8').max
    audio /= uint8_max_value

    audio *= 2.
    audio -= 1.

    return audio


def wav_to_float(audio):
    try:
        max_value = np.iinfo(audio.dtype).max
        min_value = np.iinfo(audio.dtype).min
    except:
        max_value = np.finfo(audio.dtype).max
        min_value = np.iinfo(audio.dtype).min

    audio = audio.astype('float32')
    audio -= min_value
    audio /= ((max_value - min_value) / 2.)
    audio -= 1.
    return audio

def float_to_wav(audio):
    max_value = 2147483647
    min_value = -2147483648

    audio +=1
    audio *= ((max_value - min_value) / 2.)
    audio += min_value

    return audio
