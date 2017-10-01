import numpy as np
import htkmfc as htk

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score
import ConfigParser
import logging
import time
import sys
import os

def loadData(inputData):
	featsReader = htk.open(inputData)
	trainData = featsReader.getall()
	yTrain = trainData[:, -1]
	xTrain = np.delete(trainData, -1, 1)
	del trainData
	return (xTrain, yTrain)

def cnn_reshape(X_list, windowSize):
	Y_list = []
	for i in X_list:
		j = i.reshape(windowSize,32)
		Y_list.append(j)
	Y_list = np.array(Y_list)
	Z_list = Y_list[:, np.newaxis, :, :]
	return(Z_list)

def queryWS(nameKW):
	a = {
	'government': 75,
	'company': 71,
	'hundred': 59,
	'nineteen': 79,
	'thousand': 77,
	'morning': 69,
	'business': 81
	}
	return a[nameKW]

if __name__ == "__main__":

	windowSize = queryWS(sys.argv[1])
	pathToData = '/home/rajathk/spkVer/code/dataGen/spkData/'+sys.argv[1]+'_'+str(windowSize)+'/kwsTest'
	dataFolder = 'posteriors'
	os.makedirs('posteriors')
	model = load_model('kws.h5')
	scaler = joblib.load('scaler.save')

	for dataName in os.listdir(pathToData):
		(X_val, Y_val) = loadData(pathToData+ '/' + dataName)
		X_val = scaler.transform(X_val) 
		X_val = cnn_reshape(X_val, int(windowSize))
		Y_val = np_utils.to_categorical(Y_val, 2)
		Y_predicted = model.predict(X_val, verbose = 0)
		finalPrior = np.concatenate((Y_predicted, Y_val), axis=1)
		np.savetxt(dataFolder +'/'+dataName[:-3]+'txt', finalPrior)

	windowSize = 11
	window = np.ones(windowSize)/float(windowSize)
	truthPosteriors = []
	predPosteriors = []

	for i in os.listdir('posteriors'):
		filePosterior = np.loadtxt('posteriors/'+i)
		groundTr = filePosterior[:,3]
		est = filePosterior[:,1]
		estConvolve = np.convolve(est,  window, 'same')
		truthPosteriors.append(max(groundTr))
		predPosteriors.append(max(estConvolve))

	print('Done')
	auc = roc_auc_score(truthPosteriors, predPosteriors)
	print('area under curve: ', auc)
	far, tar, thr = roc_curve(truthPosteriors, predPosteriors)
	far = far*100
	frr = (1 - tar)*100
	minDiff = min([ abs(far[i] - frr[i]) for i in range(len(far)) ])

	for i in range(len(far)):
		if abs(far[i] - frr[i]) == minDiff:
			eer = (far[i]+frr[i])/2
			print('EER: ', eer)
			break