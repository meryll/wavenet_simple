from scipy.io import wavfile
from src.utils import audio_utils, array_utils
import numpy as np
from src import settings
np.set_printoptions(threshold=np.nan)

def get_file(path):
    sample_rate, audio = read_file(path=path)
    audio = audio_utils.ensure_mono(audio=audio)

    audio = audio_utils.wav_to_float(audio=audio)
    audio= audio_utils.float_to_wav(audio=audio)
    audio = audio_utils.ensure_sample_rate(file_sample_rate=sample_rate,
                                           dest_sample_rate=settings.sample_rate,
                                           audio=audio)
    audio = audio_utils.float_to_uint8(audio=audio)

    return audio

def generate(audio):
    audio = array_utils.inverse_one_hot(audio)
    audio = audio_utils.uint8_to_float(audio=audio)
    audio = audio_utils.float_to_wav(audio=audio)
    #todo add smooth
    return np.int64(audio)

def read_file(path):
    try:
        return wavfile.read(path)
    except Exception as e:
        print(str(e))

def save(decoded):
    wavfile.write('audio/saved3.wav', data=decoded, rate=4410)

