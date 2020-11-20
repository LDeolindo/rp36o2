from pylsl import StreamInlet, resolve_stream
import time
import numpy as np
import scipy
from scipy.signal import detrend
import mne

def blink(data):
  f = open('ident.txt', 'a', encoding="utf8")

  # psd_blink, freqs_blink = mne.time_frequency.psd_array_welch(data, sfreq=250, n_per_seg=250, fmin=1, fmax=13, verbose=0)
  # psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=250, n_per_seg=250, fmin=13, fmax=100, verbose=0)
  # # retorna o index dos 2 maiores valores
  # maxValueIdx_blink = (-psd_blink).argsort()[:2]
  # maxValueIdx = (-psd).argsort()[:2]
  # # maxValueIdx = np.argmax(psd)
  # peak_blink = freqs_blink[maxValueIdx_blink[0]]
  # peak = freqs[maxValueIdx[0]]

  power = np.square(np.abs(data))
  norm_power = (power - power.min()) / (power.max() - power.min())

  thresh = 0.25

  supra_thresh = np.where(norm_power >= thresh)[0]

  sp = np.split(supra_thresh, np.where(np.diff(supra_thresh) != 1)[0] + 1)
  idx_start_end = np.array([[k[0], k[-1]] for k in sp])

  sp_amp, sp_freq = np.zeros(len(sp)), np.zeros(len(sp))

  for i in range(len(sp)):
    # Important: detrend the signal to avoid wrong peak-to-peak amplitude
    sp_amp[i] = np.ptp(detrend(data[sp[i]]))

  f.write(str(sp) + '\n')

  # if (peak_blink[0] > peak[0]):
  #   if (maxValueIdx[0] == 0 and maxValueIdx[1] == 1 or maxValueIdx[1] == 0 and maxValueIdx[0] == 1 ):
  #     f.write('\n blink \n')
  # else:
  #   f.write('1 \n')

# def blink(sampleA, sampleB):
#   f = open('identificar.txt', 'a', encoding="utf8")
#   A = np.array([0, 0, 0, 0, 0, 0, 0, 0])
#   B = np.array([0, 0, 0, 0, 0, 0, 0, 0])
#   tam = len(sampleA)
#   tam2 = len(sampleA[0])
#   for i in range(tam):
#     for j in range(tam2):
#       A[i] += sampleA[i][j]
#       B[i] += sampleB[i][j]

#   if (np.absolute(A[0] - B[0]) > 100000 and np.absolute(A[1] - B[1]) > 100000 and np.absolute(A[2] - B[2]) > 100000 and np.absolute(A[3] - B[3]) > 100000 and np.absolute(A[4] - B[4]) > 100000 and np.absolute(A[5] - B[5]) > 100000 and np.absolute(A[6] - B[6]) > 100000 and np.absolute(A[7] - B[7]) > 100000):
#     f.write('\n Jaw Clench \n')
#   elif (np.absolute(A[0] - B[0]) > 100000 and np.absolute(A[1] - B[1]) > 100000 and np.absolute(A[2] - B[2]) < 100000):
#     f.write('\n Blink \n')
#   else:
#     f.write('0 \n')


def maxAlfa(data):
  f = open('rp360.txt', 'a', encoding="utf8")
  bandPower = np.array([0, 0, 0, 0])

  # delta, _ = mne.time_frequency.psd_welch(data, n_per_seg=250 ,fmin=0.5, fmax=4)
  theta, _ = mne.time_frequency.psd_welch(data, n_per_seg=250 ,fmin=4, fmax=8)
  alfa, _ = mne.time_frequency.psd_welch(data, n_per_seg=250 ,fmin=8, fmax=13)
  beta, _ = mne.time_frequency.psd_welch(data, n_per_seg=250 ,fmin=13, fmax=32)
  gamma, _ = mne.time_frequency.psd_welch(data, n_per_seg=250 ,fmin=32, fmax=100)

  # bandPower[0] = np.average(delta)
  bandPower[0] = np.average(theta)
  bandPower[1] = np.average(alfa)
  bandPower[2] = np.average(beta)
  bandPower[3] = np.average(gamma)

  # retorna o index dos 2 maiores valores
  maxValueIdx = (-bandPower).argsort()[:2]

  # verifica se o alfa tem o maior valor
  if (maxValueIdx[0] == 1):
    maxValue = bandPower[maxValueIdx[0]]
    minValue = bandPower[maxValueIdx[1]]
    print('--------------------------------')
    print(minValue*100/maxValue)
    f.write('\n Ritmos Alpha \n')
    f.write(str(minValue*100/maxValue))
    print('--------------------------------')
  elif (maxValueIdx[0] == 2):
    maxValue = bandPower[maxValueIdx[0]]
    minValue = bandPower[maxValueIdx[1]]
    print('--------------------------------')
    print(minValue*100/maxValue)
    f.write('\n Ritmos Beta \n')
    f.write(str(minValue*100/maxValue))
    print('--------------------------------')
  else:
    f.write('0 \n')

def main():
  sample_rate = 250
  # info = mne.create_info(ch_names=['O1', 'O2', 'P3', 'Pz', 'P4', 'C3', 'Cz', 'C4'], sfreq=sample_rate, ch_types='eeg')
  info = mne.create_info(ch_names=['1', '2', '3', '4', '5', '6', '7', '8'], sfreq=sample_rate, ch_types='eeg')

  print("Looking for an EEG stream...")
  streams = resolve_stream('type', 'EEG')

  # create a new inlet to read from the stream
  inlet = StreamInlet(streams[0])

  while True:
    # sample1, _ = inlet.pull_chunk(timeout=8, max_samples=250) #
    sample1, _ = inlet.pull_chunk(timeout=1, max_samples=250) # timeout=3
    # sample2, _ = inlet.pull_chunk(timeout=2.7, max_samples=250) #
    # sample3, _ = inlet.pull_chunk(timeout=2.7, max_samples=250) #

    sample1 = np.array(sample1)
    print(len(sample1))
    # sample2 = np.array(sample2)
    # sample3 = np.array(sample3)

    # sample1 = np.concatenate((sample1, sample2, sample3))
    sample = np.transpose(sample1)

    raw = mne.io.RawArray(sample, info)
    bandPass = raw.filter(l_freq=1, h_freq=50, verbose=False)
    bandPass = bandPass.filter(l_freq=1, h_freq=50, verbose=False)
    bandPass = bandPass.filter(l_freq=1, h_freq=50, verbose=False)
    bandPass = bandPass.filter(l_freq=1, h_freq=50, verbose=False)
    maxAlfa(bandPass)

    # lowPass = raw.filter(l_freq=1, h_freq=15)
    # lowPass = lowPass.filter(l_freq=1, h_freq=15)
    # lowPass = lowPass.filter(l_freq=1, h_freq=15)
    # lowPass = lowPass.filter(l_freq=1, h_freq=15)

    # blink(sample)

if __name__ == "__main__":
  main()