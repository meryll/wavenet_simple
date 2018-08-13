import numpy as np
from src.utils import array_utils
from src import settings

def get(audio):
    frame_size = settings.frame_size
    frame_shift=settings.frame_shift

    audio = audio[:10000]
    audio_len = len(audio)
    X = []
    y = []

    for i in range(0, audio_len - frame_size - 1-frame_size, frame_shift):

        frame = audio[i:i + frame_size]
        if len(frame) < frame_size:
            break
        if i + frame_size >= audio_len:
            break

        temp = audio[i + frame_size:i+frame_size+frame_size]

        X.append(array_utils.one_hot(frame))
        y.append(array_utils.one_hot(temp))

    return np.array(X),np.array(y)