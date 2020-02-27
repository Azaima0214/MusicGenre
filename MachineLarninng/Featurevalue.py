#!/usr/bin/env python
# coding: utf-8

# In[6]:


import librosa
import numpy as np
import os
a=[]
b=[3.78481264e-01,1.40893798e-01,2.19767792e+03,2.24740944e+03,4.56779383e+03,1.03037465e-01,-1.36793235e+02,9.94826654e+01,-8.90716924e+00,3.58347341e+01,-5.32793760e-01,1.42701158e+01,-4.66991523e+00,9.85852476e+00,-6.48586136e+00,7.61237174e+00,-5.72946021e+00,4.50112703e+00,-4.59273680e+00,1.84468859e+00,-3.69927087e+00,1.35565453e+00,-3.94201266e+00,6.71545229e-01,-2.35990120e+00,-8.04243081e-01]
c=[8.00704312e-02,7.53247096e-02,6.94102707e+02,5.07779798e+02,1.52006735e+03,4.12784390e-02,1.01823417e+02,3.04530080e+01,2.08780689e+01,1.59584665e+01,1.20822892e+01,1.15307248e+01,9.75954633e+00,1.01351807e+01,8.11058129e+00,7.70706334e+00,6.57979511e+00,6.49638469e+00,5.96607735e+00,4.96029510e+00,4.85842088e+00,4.39119570e+00,4.48102461e+00,3.79944707e+00,3.62791300e+00,3.88781302e+00]
y, sr = librosa.load('/Users/k17070kk/Downloads/audio_get/Blues/Blues_2.m4a', mono=True, duration=30)
a.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
a.append(np.mean(librosa.feature.rmse(y=y)))
a.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
a.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
a.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
a.append(np.mean(librosa.feature.zero_crossing_rate(y)))
mfcc = librosa.feature.mfcc(y=y, sr=sr)
for e in mfcc:
    a.append(np.mean(e))
print(a[1])
for i in range(0,26):
        a[i]=(a[i]-b[i])/c[i]
print(a)

