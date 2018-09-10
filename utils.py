import random
import librosa
import python_speech_features as psf
import numpy as np
import os
import keras
import os.path
import csv
from tqdm import tqdm




def distinguishDataset(filePath, testPer=0.0):
    wavLabel = []
    wavPath = []
    trainWavLabel = []
    trainwavPath = []
    testWavLabel = []
    testwavPath = []
    categoryNum = len(os.listdir(filePath))
    wavNum = 0
    for index, cate in tqdm(enumerate(os.listdir(filePath))):
        path = os.path.join(filePath, cate)
        for i in range(len(os.listdir(path))):
            wavDir = os.path.join(path, os.listdir(path)[i])
            wavLabel.append(index)
            wavPath.append(wavDir)
            wavNum +=1
    print("读取完成")
    c = list(zip(wavLabel, wavPath))
    random.shuffle(c)
    wavLabel[:], wavPath[:] = zip(*c)

    testWavLabel = wavLabel[0:int(wavNum*testPer)]
    testwavPath = wavPath[0:int(wavNum*testPer)]
    trainWavLabel = wavLabel[int(wavNum*testPer):]
    trainwavPath = wavPath[int(wavNum*testPer):]

    f = open("train.csv", "w", newline='')
    writer = csv.writer(f)
    for i in tqdm(range(len(trainWavLabel))):
        writer.writerow([trainwavPath[i], trainWavLabel[i]])
    f.close()
    print("写入train,csv完成 ")
    f = open("test.csv", "w", newline='')
    writer = csv.writer(f)
    for i in tqdm(range(len(trainWavLabel))):
        writer.writerow([trainwavPath[i], trainWavLabel[i]])
    f.close()
    print("写入test.csv完成 ")


def getBatchDataset(batchSize, csvFile, nClass):
    while True:
        csvData = csv.reader(open(csvFile, "r"))
        batchData = []
        batchLabel = []
        count = 0
        for data in csvData:
            # print(count)
            d = data[0]
            label = data[1]
            # print(data)
            signal, sampleRate = librosa.load(d, sr=16000)
            if len(signal)>=16000:
                signal = signal[len(signal)-16000:]
            else:
                continue

            fbank = psf.logfbank(signal, samplerate=sampleRate)
            delta1 = psf.delta(fbank, 1)
            delta2 = psf.delta(fbank, 2)

            feat = fbank.T[:, :, np.newaxis] # (26, 99, 1)
            feat1 = delta1.T[:, :, np.newaxis] # (26, 99, 1)
            feat2 = delta2.T[:, :, np.newaxis] # (26, 99, 1)
            # print(feat.shape)
            # print(feat1.shape)
            # print(feat2.shape)

            mergeFbank = np.concatenate((feat, feat1, feat2), axis=2)
            label = keras.utils.to_categorical(label, nClass)

            batchData.append(mergeFbank)
            batchLabel.append(label)
            count +=1

            if count==batchSize:
                x = batchData
                y = batchLabel
                batchData = []
                batchLabel = []
                count = 0
                print(y)
                print(x)
                # return [np.array(x), np.array(y)]
                yield [np.array(x), np.array(y)]


if __name__ == '__main__':
    filePath = "data"
    distinguishDataset(filePath, testPer=0.05)
