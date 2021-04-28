#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import Queue
import math
import copy
import wave
import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as pl
import scipy.io.wavfile as wf

baud_rate = 100
data_length = 12
frame_size = 480
sample_rate = 48000
out_ample = 10000
mask_audio = "mask.wav"


"""
精度就是fs/N，所以点数约大，fs/N越小，也就是精度越高。
补0到2的幂次方个点可以提高fft的速度，但无法提高fft精度。
"""
f0 = 152 * 93.75  # 14250
f1 = 156 * 93.75  # 14625
f2 = 160 * 93.75  # 15000
f3 = 164 * 93.75  # 15375
f4 = 168 * 93.75  # 15750


def ploy_data(y):
    pl.specgram(y, NFFT=512, Fs=48000, noverlap=480)
    pl.grid(True)
    pl.show()


'''
return ascii value
'''
def CharToDec(ori_data="NC_5F_CENTER_5G:hellohello"):
    dec_data = []
    # ori_data = ori_data.encode()
    print("Raw data: %s" % ori_data)
    for data in ori_data:
        dec_data.append(ord(data))
    return dec_data


'''
crc encode one byte
'''
def CalCrcOneByte(abyte):
    crc_1byte = 0
    for i in range(8):
        if (crc_1byte ^ abyte) & 0x01:
            crc_1byte = crc_1byte ^ 0x18
            crc_1byte = crc_1byte >> 1
            crc_1byte = crc_1byte | 0x80
        else:
            crc_1byte = crc_1byte >> 1
        abyte = abyte >> 1
    return crc_1byte


'''
crc encode n bytes
'''
def CalCrcNbyte(data):
    crc = 0
    for i in range(len(data)):
        crc = CalCrcOneByte(crc ^ data[i])
    return crc


'''
0xAA(1 byte) + Data长度n(1 byte) + (包数 + 包号)(1 byte) + Data(n bytes)
'''
def DivideData(data, q):
    length = len(data)
    temp1 = length % data_length
    temp2 = length / data_length

    if temp1 == 0:
        package_num = temp2
    else:
        package_num = temp2 + 1

    for i in range(temp2):
        d = data[i * data_length: (i + 1) * data_length]
        d.insert(0, (package_num << 4 | i))
        d.insert(0, data_length)
        d.insert(0, 0xaa)
        q.put(d)
        print("package %d:" % i, d)

    if temp2 == 0:
        i = -1

    if temp1 != 0:
        d = data[(i + 1) * data_length:]
        d.insert(0, (package_num << 4 | (i + 1)))
        d.insert(0, len(d) - 1)
        d.insert(0, 0xaa)
        q.put(d)
        print("package %d:" % (i + 1), d)


def EncodeOneBit(fb):
    freqArray = np.hstack(np.multiply(np.ones(frame_size), fb))
    timeArray = np.arange(0, 1.0 / float(baud_rate), 1.0 / float(sample_rate), dtype=np.float)
    bitArray = out_ample * np.cos(2 * np.pi * np.multiply(freqArray[:frame_size], timeArray[:frame_size]))
    bitArray = bitArray * np.hanning(frame_size)  # * np.hanning(frame_size)
    return bitArray


'''
两个bit编码为480个点, 即为一个Frame
'''
def EncodeOneFrame(bit0, bit1, syncFlag):
    frameArray = np.zeros(frame_size)
    if bit0 == 0:
        frameArray = np.add(frameArray, EncodeOneBit(f1))
    else:
        frameArray = np.add(frameArray, EncodeOneBit(f2))
    if bit1 == 0:
        frameArray = np.add(frameArray, EncodeOneBit(f3))
    else:
        frameArray = np.add(frameArray, EncodeOneBit(f4))
    if syncFlag == 1:
        frameArray = np.add(frameArray, EncodeOneBit(f0))
    return frameArray


'''
编码一个byte的数据
'''
def EncodeOneByte(byte, syncFlag):
    i = 0
    byteArray = []
    while i < 4:
        bit0 = (byte & 0x80) >> 7
        bit1 = (byte & 0x40) >> 6
        byteArray.extend(EncodeOneFrame(bit0, bit1, syncFlag))
        syncFlag = 0
        byte <<= 2
        i += 1
    return byteArray


'''
编码所要发送的数据:用户名和密码, n bytes
'''
def EncodeData(data):
    i = 0
    dataArray = []
    syncFlag = 1

    for num in data:
        dataArray.extend(EncodeOneByte(num, syncFlag))
        syncFlag = 0
    return dataArray


def LowpassFilter(wave_data):
    b = signal.remez(201, (0, 0.18, 0.21, 0.5), (1, 0.001))
    return signal.lfilter(b, [1], wave_data)


'''
打开所要叠加的音频文件，即掩盖音文件
'''
def AddMaskAudio(data, audio_name):
    f = wave.open(audio_name, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    # print("nchannels:", nchannels)
    # print("sampwidth:", sampwidth)
    # print("framerate:", framerate)
    # print("  nframes:", nframes)

    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.int16)

    '''
    双通道数据转为单通道数据
    '''
    if nchannels == 2:
        wave_data.shape = -1, 2
        wave_data = wave_data.T[0]

    wave_data = LowpassFilter(wave_data)

    i = 0
    datalength = len(data)
    if datalength > len(wave_data):
        while i < datalength:
            data[i] += 0.25 * wave_data[i % nframes]
            i = i + 1
        return data
    else:
        wavelength = len(wave_data) / frame_size * frame_size;
        while i < wavelength:
            wave_data[i] = 0.25 * wave_data[i] + data[i % datalength]
            i = i + 1
        return wave_data


def AddAudioData(y):
    y = np.array(y)
    filename = "audio" + ".wav"
    y = AddMaskAudio(y, mask_audio)

    # ploy_data(y)
    b = len(y) / frame_size
    wf.write("./{0}".format(filename), len(y) / b * baud_rate, y.astype(np.dtype('i2')))


if __name__ == "__main__":

    dataQueue = Queue.Queue()

    if len(sys.argv) == 1:
        DecData = CharToDec()
    elif len(sys.argv) == 2:
        DecData = CharToDec(sys.argv[1])
    elif len(sys.argv) == 3:
        DecData = CharToDec(sys.argv[1])
        mask_audio = sys.argv[2]

    DivideData(DecData, dataQueue)
    gap = np.zeros(480 * 4).astype("i2")

    i = 0
    dataArray = []
    while not dataQueue.empty():
        q = dataQueue.get()
        q.append(CalCrcNbyte(q))
        print("add crc", q)
        dataArray.extend(EncodeData(q))
        dataArray.extend(gap)
        i = i + 1
    ploy_data(dataArray)

    '''
    手机播放音频的时候有淡入淡出效果, 所以开头多添加两个字节的数据
    '''
    dataTransmit = []
    head = np.zeros(480 * 8).astype("i2")
    dataTransmit.extend(head)

    '''
    一个wave文件重复播放4次完整的用户名和密码数据
    '''
    for i in range(4):
        dataTransmit.extend(dataArray)

    AddAudioData(dataTransmit)
