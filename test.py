from __future__ import print_function
import scipy
import numpy as np





from matplotlib import pyplot as plt
from scipy.fftpack import fft
import scipy.io.wavfile as wavfile

#import wave
#wave_file = wave.open("Penguin.wav")
#freq = wave_file.getframerate()
#print(freq)

fs, data = wavfile.read('Penguin.wav')      # load the data

a = data.T[0]                               # this is a two channel soundtrack, I get the first track
#print(a)

b=[(ele/2**8.)*2-1 for ele in a]            # this is 8-bit track, b is now normalized on [-1,1)
#print(b)

c = fft(b)                                  # calculate fourier transform (complex numbers list)
#print(c)

d = len(c)/2                                # you only need half of the fft list (real signal symmetry)
print(d)


plt.plot(abs(c[:(int(d)-1)]), 'r')
plt.show()