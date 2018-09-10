import numpy as np
import keras
import librosa
import python_speech_features as psf
import csv
from utils import getBatchDataset




if __name__ == '__main__':

    for i in getBatchDataset(1, "test.csv", 8):
        print(1)
        print(i)