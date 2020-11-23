from pylsl import StreamInlet, resolve_stream
import time
import numpy as np
import scipy
from scipy.signal import detrend
import mne


def blink(previousData, data):
    previousData = np.array(previousData[0])
    data = np.array(data[0])

    absPreviousData = np.absolute(previousData[0])
    absData = np.absolute(data[0])

    previousMagnitude = np.linalg.norm(absPreviousData)
    dataMagnitude = np.linalg.norm(absData)
    print(previousMagnitude, dataMagnitude)

    resultDiffMagnitudes = (previousMagnitude * 100) / dataMagnitude

    if resultDiffMagnitudes >= 25:
        print('BLINK ROLOU')
    else:
        print('NÃO DEU')

    if dataMagnitude >= previousMagnitude * 1.25:
        print('BLINK ROLOU 2')
    else:
        print('NÃO DEU 2')



def findEvents(data):
    f = open('rp36o.txt', 'a', encoding="utf8")
    bandPower = np.array([0, 0, 0, 0, 0])

    delta, _ = mne.time_frequency.psd_welch(
        data, n_per_seg=250, fmin=1, fmax=4)
    theta, _ = mne.time_frequency.psd_welch(
        data, n_per_seg=250, fmin=4, fmax=8)
    alfa, _ = mne.time_frequency.psd_welch(
        data, n_per_seg=250, fmin=8, fmax=13)
    beta, _ = mne.time_frequency.psd_welch(
        data, n_per_seg=250, fmin=13, fmax=32)
    gamma, _ = mne.time_frequency.psd_welch(
        data, n_per_seg=250, fmin=32, fmax=100)

    bandPower[0] = np.average(delta)
    bandPower[1] = np.average(theta)
    bandPower[2] = np.average(alfa)
    bandPower[3] = np.average(beta)
    bandPower[4] = np.average(gamma)

    # returns an array with the index of the 2 highest values
    maxValueIdx = (-bandPower).argsort()[:2]

    # checks if the Alpha has the highest value
    if (maxValueIdx[0] == 2):
        maxValue = bandPower[maxValueIdx[0]]
        minValue = bandPower[maxValueIdx[1]]
        f.write('\n Ritmos Alpha \n')
        f.write(str(minValue*100/maxValue))
    # checks if the Beta has the highest value
    elif (maxValueIdx[0] == 3):
        maxValue = bandPower[maxValueIdx[0]]
        minValue = bandPower[maxValueIdx[1]]
        f.write('\n Ritmos Beta \n')
        f.write(str(minValue*100/maxValue))
    else:
        f.write('0 \n')


def main():
    sample_rate = 250
    # info = mne.create_info(ch_names=['O1', 'O2', 'P3', 'Pz', 'P4', 'C3', 'Cz', 'C4'], sfreq=sample_rate, ch_types='eeg')
    info = mne.create_info(ch_names=[
                           '1', '2', '3', '4', '5', '6', '7', '8'], sfreq=sample_rate, ch_types='eeg')

    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    previousLowPass = None
    execution = 0

    while True:
        # sample1, _ = inlet.pull_chunk(timeout=8, max_samples=250) #
        
        sample1, _ = inlet.pull_chunk(
            timeout=1.0, max_samples=250)  # timeout=3 / timeout=1.5
        sample2, _ = inlet.pull_chunk(timeout=1.0, max_samples=250)
        sample3, _ = inlet.pull_chunk(timeout=1.0, max_samples=250)

        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        sample3 = np.array(sample3)

        sample1 = np.concatenate((sample1, sample2, sample3))
        sample = np.transpose(sample1)

        raw = mne.io.RawArray(sample, info)

        bandPass = raw.filter(l_freq=1, h_freq=50)
        bandPass = bandPass.filter(l_freq=1, h_freq=50)
        bandPass = bandPass.filter(l_freq=1, h_freq=50)
        bandPass = bandPass.filter(l_freq=1, h_freq=50)

        findEvents(bandPass)

        lowPass = raw.filter(l_freq=1, h_freq=10)
        lowPass = lowPass.filter(l_freq=1, h_freq=10)
        lowPass = lowPass.filter(l_freq=1, h_freq=10)
        lowPass = lowPass.filter(l_freq=1, h_freq=10)

        if execution == 1:
            blink(previousLowPass, lowPass)
        
        previousLowPass = lowPass
        execution = 1


if __name__ == "__main__":
    with mne.utils.use_log_level("error"):
        main()
