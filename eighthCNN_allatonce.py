'''
This needs to be done: type activate tensorflow-gpu
http://www.heatonresearch.com/2017/01/01/tensorflow-windows-gpu.html
'''
# Simple CNN model for CIFAR-10
import datetime
import dicom
from keras import backend as K
from keras import backend as K
from keras.constraints import maxnorm
from keras.datasets import cifar10
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import SimpleRNN
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from matplotlib import pyplot, cm
import numpy as np
import os
import pandas
import progressbar
import scipy.misc
from scipy.misc import toimage
import time
import tensorflow
K.set_image_dim_ordering('th')

'''
Parameters the user can set:
'''
parentdirectory="C:\\Users\\daniel\\Documents\\Machine Learning And Statistics Projects\\LungCancer\\stage1\\stage1"
#parentdirectory="C:\\Users\\bermads1\\Downloads\\cancer data"
answersfile="..\\..\\stage1_labels.csv"
frogclassifier="C:\\Python27\\firstCNN\\vgg16_weights.h5"
modelsavename="DataBowl2017 modelLSTM_frog_classifier_"
#model=load_model("DataBowl2017 modelLSTM_frog_classifier_20170219 164611.h5")

#Set model learning parameters:
preprocessTrainingData=True #if you need to perform feature extraction on training data
preprocessTestingData=True #if you need to perform feature extraction on training data
labeledTrainingData=True
labeledTestingData=False #setting to false means you are taking all the data from the directory list that does not have a label in the answerfile
						 #If you set it to True, you must come up with a way to draw from the answer file.

#When drawing from the same file, use these 
startTrain=0
endTrain=1396
startTest=500
endTest=700

#learning parameters:
epochs=40
learning_rate = .1
decay_rate = 1e-6
momentum = 0.9




os.chdir(parentdirectory)
directoryList=[x for x in os.listdir(parentdirectory) if os.path.isdir(os.path.join(parentdirectory,x))]
print("Loading in answer file:")
answers=pandas.io.parsers.read_table(answersfile,',')
print("Answer file loaded.")



print("Generating training and test data.")
#Training data
solutions=answers.iloc[startTrain:(endTrain+1)]
numberofTrains=solutions.shape[0]
#Test Data
#validationList=answers.iloc[startTest:(endTest+1)]
#numberofTests=validationList.shape[0]
#validationList=pandas.DataFrame({'id':validationList['id']})
validationList=list(set(directoryList)-set(answers['id']))
numberofTests=len(validationList)
validationList=pandas.DataFrame({'id':validationList})

'''
Parameters not advised to touch.
'''
#Set image size. This is determined by the frog classifier and if it is changed, the frog classifier has to be retrained.
IMG_PX_SIZE=224
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
mostfiles=0


#block of code to find the longest model sequence
for direc in answers['id']:
	files=[f for f in os.listdir(direc) if (os.path.isfile(os.path.join(direc,f)) and ".dcm" in f.lower())]
	if(len(files)>mostfiles):
		mostfiles=len(files)


def BuildFiles(ids, numberofTrials, IMG_PX_SIZE, featureExtractor, X_ting, labeledData, y_ting=None):
	'''
	ids: a pandas.DataFrame that has two columns 'cancer' and 'ids'. It won't have cancer column if you do not know the labels.
		Cancer tells you if it's cancer (1) or not (0) 
		ids tells you the subject id with the folder containing all images.
	numberofTrials: tells you how many subjects there are
	IMG_PX_SIZE: how large the images are. Must be a square image so one integer
	featureExtractor: is the keras model that is applied to get the features from the image
	labeledData: is a boolean indicating whether or not the data is labeled
	X_training: a predefined np array that has dimensions numberofTrials, mostfiles, 4096
	y_training: a predefined np array that has dimensions numberoftrials, 1
	
	'''
	pbar = progressbar.ProgressBar(maxval=1).start()
	for direc,k in zip(ids['id'],range(0,numberofTrials)):
		X_t=[]
		files=list(os.path.join(direc,f) for f in os.listdir(direc) if (os.path.isfile(os.path.join(direc,f)) and ".dcm" in f.lower()))
		if labeledData:
			y_t=ids['cancer'][ids['id']==direc]
			y_t=y_t.iloc[0]
			y_t=np.array([y_t])
		lenfiles=sum(1 for x in files) 
		features=np.zeros((lenfiles,1,4096))
		ds=[dicom.read_file(file) for file in files]
		ds.sort(key=lambda x: int(x.ImagePositionPatient[2]))
		img=[normalize(np.array(d.pixel_array)) for d in ds]
		img=[scipy.misc.imresize(im,(IMG_PX_SIZE,IMG_PX_SIZE)) for im in img]
		reshapedimg=list(np.reshape(im,(1,IMG_PX_SIZE,IMG_PX_SIZE)) for im in img)
		X_t=list(np.concatenate((reshapedim,reshapedim,reshapedim),axis=0) for reshapedim in reshapedimg)
		features=np.asarray([featureExtractor.predict([X.reshape(1,3,IMG_PX_SIZE,IMG_PX_SIZE)]) for X in X_t])
		if labeledData:
			y_ting[k][:]=np.array(y_t)
		X_ting[k][(mostfiles-len(files)):mostfiles,:]=features.reshape((len(files),4096))
		pbar.update(k/numberofTrials)
	pbar.finish()
	if labeledData:
		return X_ting, y_ting
	else:
		return X_ting

def normalize(image):
	#function that normalizes the data. The bounds are set prior.
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
	
def CreateCNN(vgg16):
	featureExtractor = Sequential()
	featureExtractor.add(ZeroPadding2D((1,1),input_shape=(3,IMG_PX_SIZE, IMG_PX_SIZE)))
	featureExtractor.add(Convolution2D(64, 3, 3, activation='relu',	weights=vgg16.layers[1].get_weights()))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(64, 3, 3, activation='relu',	weights=vgg16.layers[3].get_weights()))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(128, 3, 3, activation='relu', weights=vgg16.layers[6].get_weights()))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(128, 3, 3, activation='relu', weights=vgg16.layers[8].get_weights()))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(256, 3, 3, activation='relu', weights=vgg16.layers[11].get_weights()))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(256, 3, 3, activation='relu',	weights=vgg16.layers[13].get_weights()))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(256, 3, 3, activation='relu',	weights=vgg16.layers[15].get_weights()))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu',	weights=vgg16.layers[18].get_weights()))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu',	weights=vgg16.layers[20].get_weights()))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu',	weights=vgg16.layers[22].get_weights()))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu',	weights=vgg16.layers[25].get_weights()))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu',	weights=vgg16.layers[27].get_weights()))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu',	weights=vgg16.layers[29].get_weights()))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))
	featureExtractor.add(Flatten())
	featureExtractor.add(Dense(4096, activation='relu',	weights=vgg16.layers[32].get_weights()))

	return featureExtractor
	
def VGG_16(fullmodel):	
	'''
	builds the VGG_16 classifier 
	'''
	
	featureExtractor = Sequential()
	featureExtractor.add(ZeroPadding2D((1,1),input_shape=(3,IMG_PX_SIZE, IMG_PX_SIZE)))
	featureExtractor.add(Convolution2D(64, 3, 3, activation='relu'))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(64, 3, 3, activation='relu'))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(128, 3, 3, activation='relu'))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(128, 3, 3, activation='relu'))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(256, 3, 3, activation='relu'))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(256, 3, 3, activation='relu'))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(256, 3, 3, activation='relu'))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu'))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu'))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu'))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu'))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu'))
	featureExtractor.add(ZeroPadding2D((1,1)))
	featureExtractor.add(Convolution2D(512, 3, 3, activation='relu'))
	featureExtractor.add(MaxPooling2D((2,2), strides=(2,2)))

	featureExtractor.add(Flatten())
	featureExtractor.add(Dense(4096, activation='relu'))
	featureExtractor.add(Dropout(0.5))
	featureExtractor.add(Dense(4096, activation='relu'))
	featureExtractor.add(Dropout(0.5))
	featureExtractor.add(Dense(1000, activation='softmax'))

	featureExtractor.load_weights(fullmodel)
	return featureExtractor

	
'''
These lines of code load in the VGG_16 model and then cut off the last few decision layers
'''
vgg16=VGG_16(frogclassifier)
featureExtractor=CreateCNN(vgg16)

#Define the LSTM model
model = Sequential()
model.add(LSTM(32,return_sequences=False,stateful=False,input_shape=(mostfiles,4096)))
model.add((Dense(1, activation='sigmoid')))
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# numberofTests=200

#block of code to save the model
ts=time.time()
st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")
filename=modelsavename+st

numberofSubTrains=int(len(solutions)/2)
numberofSubTrains2=int(int(len(solutions)-numberofSubTrains)/2+numberofSubTrains)
solutions1=solutions.iloc[0:numberofSubTrains]
solutions2=solutions.iloc[numberofSubTrains:numberofSubTrains2]
solutions3=solutions.iloc[numberofSubTrains2:(numberofTrains)]
	
# if preprocessTrainingData:
	# numberofTrains1=numberofSubTrains
	# X_training1=np.zeros((numberofSubTrains,mostfiles,4096))
	# y_training1=np.zeros((numberofSubTrains,1))
	
	# #If you are saving the data
	# X_training1, y_training1 = BuildFiles(solutions1, numberofSubTrains, IMG_PX_SIZE, featureExtractor, X_training1, labeledTrainingData, y_training1)	
	# np.save("X_training0to1397VGG16pt1.npz",X_training1)
	# np.save("y_training0to1397VGG16pt1.npz",y_training1)
# else:
	# #If you are loading the data
	# X_training1=np.load("X_training0to1397VGG16pt1.npz.npy")
	# y_training1=np.load("y_training0to1397VGG16pt1.npz.npy")
if preprocessTrainingData:
	numberofTrains2=len(solutions2.index)
	print(numberofTrains2)
	print(mostfiles)
	X_training2=np.zeros((numberofTrains2,mostfiles,4096))
	y_training2=np.zeros((numberofTrains2,1))
	#If you are saving the data
	X_training2, y_training2 = BuildFiles(solutions2, numberofTrains2, IMG_PX_SIZE, featureExtractor, X_training2, labeledTrainingData, y_training2)	
	np.save("X_training0to1397VGG16pt2.npz",X_training2)
	np.save("y_training0to1397VGG16pt2.npz",y_training2)
else:
	#If you are loading the data
	X_training2=np.load("X_training0to1397VGG16pt2.npz.npy")
	y_training2=np.load("y_training0to1397VGG16pt2.npz.npy")

if preprocessTrainingData:
	numberofTrains3=len(solutions3.index)
	
	X_training3=np.zeros((numberofTrains3,mostfiles,4096))
	y_training3=np.zeros((numberofTrains3,1))
	#If you are saving the data
	X_training3, y_training3 = BuildFiles(solutions3, numberofTrains3, IMG_PX_SIZE, featureExtractor, X_training3, labeledTrainingData, y_training3)	
	np.save("X_training0to1397VGG16pt3.npz",X_training3)
	np.save("y_training0to1397VGG16pt3.npz",y_training3)
else:
	#If you are loading the data
	X_training3=np.load("X_training0to1397VGG16pt3.npz.npy")
	y_training3=np.load("y_training0to1397VGG16pt3.npz.npy")

		
if preprocessTestingData:
	X_testing=np.zeros((numberofTests,mostfiles,4096))
	y_testing=np.zeros((numberofTests,1))
	if labeledTestingData:
		X_testing, y_testing = BuildFiles(validationList, numberofTests, IMG_PX_SIZE, featureExtractor, X_testing, labeledTestingData, y_testing)	
		np.save("y_testing500to700VGG16.npz",y_testing)
	if not labeledTestingData:
		X_testing = BuildFiles(validationList, numberofTests, IMG_PX_SIZE, featureExtractor, X_testing, labeledTestingData, y_testing)	
	np.save("X_validationVGG16.npz",X_testing)
else:
	X_testing=np.load("X_Verification.npz.npy")
	if labeledTestingData:
		y_testing=np.load("y_testing500to700.npz.npy")

# X_training=np.concatenate((X_training,X_training),axis=0)
# y_training=np.concatenate((y_training,y_training),axis=0)
model.fit(X_training, y_training,nb_epoch=epochs)
predictions=model.predict(X_testing)

print(predictions)
results=pandas.DataFrame({'id':pandas.Series(validationList),'cancer':pandas.Series(predictions)})

results.to_csv("cancerpredictions.csv",index=False)
#scores = model.evaluate(X_testing, y_testing, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
