import numpy as np
import librosa
import os
from keras import models
from keras import layers

class predict_music():

	predict_data = []
	model = ""
	labels = ['Blues','Classical','Country','Disco','HipHop','Jazz','Metal','POP','Reggae','Rock','Alternative']
	mean =  [3.78481264e-01,1.40893798e-01,2.19767792e+03,2.24740944e+03,4.56779383e+03,1.03037465e-01,-1.36793235e+02,9.94826654e+01,-8.90716924e+00,3.58347341e+01,-5.32793760e-01,1.42701158e+01,-4.66991523e+00,9.85852476e+00,-6.48586136e+00,7.61237174e+00,-5.72946021e+00,4.50112703e+00,-4.59273680e+00,1.84468859e+00,-3.69927087e+00,1.35565453e+00,-3.94201266e+00,6.71545229e-01,-2.35990120e+00,-8.04243081e-01]

	std_diviation = [8.00704312e-02,7.53247096e-02,6.94102707e+02,5.07779798e+02,1.52006735e+03,4.12784390e-02,1.01823417e+02,3.04530080e+01,2.08780689e+01,1.59584665e+01,1.20822892e+01,1.15307248e+01,9.75954633e+00,1.01351807e+01,8.11058129e+00,7.70706334e+00,6.57979511e+00,6.49638469e+00,5.96607735e+00,4.96029510e+00,4.85842088e+00,4.39119570e+00,4.48102461e+00,3.79944707e+00,3.62791300e+00,3.88781302e+00]
	
	def load_model(self):
		self.model = models.load_model('/home/dev/DnoIshi/data/model.h5', compile=False)
		
	def load_file(self, filename):
		self.predict_data = []
		y, sr = librosa.load("/home/dev/DnoIshi/data/" + filename)
		
		self.predict_data.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
		self.predict_data.append(np.mean(librosa.feature.rmse(y=y)))
		self.predict_data.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
		self.predict_data.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
		self.predict_data.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
		self.predict_data.append(np.mean(librosa.feature.zero_crossing_rate(y)))
		mfcc = librosa.feature.mfcc(y=y, sr=sr)
		
		for e in mfcc:
			self.predict_data.append(np.mean(e))
		for i in range(0,26):
			self.predict_data[i] = ( self.predict_data[i] - self.mean[i] ) / self.std_diviation[i]
	
	def predict_genre(self):
		data = np.array(self.predict_data).reshape(1, 26)
		result = self.model.predict_on_batch(data)[0]
		return self.labels[result.argmax()]
	
	def __init__(self):
		self.load_model()