#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import wave
from python_speech_features import mfcc
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from dtw import *
from numpy.linalg import norm
#from dtaidistance import dtw
#from dtaidistance import dtw_visualisation as dtwvis


def predict_animal(test_file_name):
    y2, sr2 = librosa.load(test_file_name)
    mfcc2 = librosa.feature.mfcc(y2, sr2)
    names = ["birds","cat","cow","dog","elephant","horse","lion","penguin"]
    num_clips = [4,4,5,4,3,3,4,10]
    min_dist = []

    for i in range(len(names)):
        curr_dist_vec = []
        for j in range(1,num_clips[i]+1):
            curr_file_name = names[i]+str(j)+".wav"
            y1, sr1 = librosa.load(curr_file_name)
            mfcc1 = librosa.feature.mfcc(y1,sr1)
            dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x-y, ord=1))
            print("The normalized distance between " + test_file_name + " and " + curr_file_name +
                  " is: " + str(dist))   # 0 for similar audios
            curr_dist_vec.append(dist)
        print("here's the minimum distance between " + test_file_name + " and " + 
             names[i] + ": " + str(np.min(curr_dist_vec)))
        min_dist.append(np.min(curr_dist_vec))


    print("Here's the match: " + str(names[min_dist.index(np.min(min_dist))])) 

