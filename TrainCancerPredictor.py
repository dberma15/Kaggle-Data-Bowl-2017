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
from keras.layers.convolutional import MaxPooling1D
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
import pickle
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
answersfile="..\\..\\stage1_labels.csv"
frogclassifier="C:\\Python27\\firstCNN\\CNNfromgrayscalecifar10data100by100.h5"
modelsavename="DataBowl2017 modelLSTM_frog_classifier_"

#Set model learning parameters:
preprocessTrainingData=False #if you need to perform feature extraction on training data
preprocessTestingData=False #if you need to perform feature extraction on training data
labeledTrainingData=False
labeledTestingData=False #setting to false means you are taking all the data from the directory list that does not have a label in the answerfile
						 #If you set it to True, you must come up with a way to draw from the answer file.

#When drawing from the same file, use these 
startTrain=0
endTrain=1396
startTest=1397
endTest=1597

#learning parameters:
epochs=50
learning_rate = .1
decay_rate = .1
momentum = 2


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
validationList=answers.iloc[startTest:(endTest+1)]
numberofTests=validationList.shape[0]
validationList=pandas.DataFrame({'id':validationList['id']})

'''
Parameters not advised to touch.
'''
#Set image size. This is determined by the frog classifier and if it is changed, the frog classifier has to be retrained.
IMG_PX_SIZE=150
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
mostfiles=0


#block of code to find the longest model sequence
answers=os.listdir()
for direc in answers:
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
	X_training: a predefined np array that has dimensions numberofTrials, mostfiles, 512
	y_training: a predefined np array that has dimensions numberoftrials, 1
	
	'''
	pbar = progressbar.ProgressBar(maxval=1).start()
	directoryOrder=[]
	for direc,k in zip(ids['id'],range(0,numberofTrials)):
		directoryOrder.append(direc)
		X_t=[]
		files=list(os.path.join(direc,f) for f in os.listdir(direc) if (os.path.isfile(os.path.join(direc,f)) and ".dcm" in f.lower()))
		if labeledData:
			y_t=ids['cancer'][ids['id']==direc]
			y_t=y_t.iloc[0]
			y_t=np.array([y_t])
		lenfiles=sum(1 for x in files) 
		features=np.zeros((lenfiles,1,512))
		ds=[dicom.read_file(file) for file in files]
		ds.sort(key=lambda x: int(x.ImagePositionPatient[2]))
		img=[normalize(np.array(d.pixel_array)) for d in ds]
		img=[scipy.misc.imresize(im,(IMG_PX_SIZE,IMG_PX_SIZE)) for im in img]
		X_t=list(np.reshape(im,(1,IMG_PX_SIZE,IMG_PX_SIZE)) for im in img)
		# X_t=list(reshapedim.reshape for reshapedim in reshapedimg)
		features=np.asarray([featureExtractor.predict([X.reshape(1,1,IMG_PX_SIZE,IMG_PX_SIZE)]) for X in X_t])
		if labeledData:
			y_ting[k][:]=np.array(y_t)
		X_ting[k][(mostfiles-len(files)):mostfiles,:]=features.reshape((len(files),512))
		pbar.update(k/numberofTrials)
	pbar.finish()
	if labeledData:
		return X_ting, y_ting
	else:
		return X_ting, directoryOrder

def normalize(image):
	#function that normalizes the data. The bounds are set prior.
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
	
def CreateCNN(vgg16):
	print(vgg16.summary())
	featureExtractor = Sequential()
	featureExtractor.add(Convolution2D(32, 3, 3, input_shape=(1, IMG_PX_SIZE, IMG_PX_SIZE), activation='relu', border_mode='same',weights=vgg16.layers[0].get_weights()))
	#featureExtractor.add(Dropout(0.2))
	featureExtractor.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same',weights=vgg16.layers[2].get_weights()))
	featureExtractor.add(MaxPooling2D(pool_size=(2, 2)))
	featureExtractor.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',weights=vgg16.layers[4].get_weights()))
	#featureExtractor.add(Dropout(0.2))
	featureExtractor.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same',weights=vgg16.layers[6].get_weights()))
	featureExtractor.add(MaxPooling2D(pool_size=(2, 2)))
	featureExtractor.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same',weights=vgg16.layers[8].get_weights()))
	#featureExtractor.add(Dropout(0.2))
	featureExtractor.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same',weights=vgg16.layers[10].get_weights()))
	featureExtractor.add(MaxPooling2D(pool_size=(2, 2)))
	featureExtractor.add(Flatten())
	#featureExtractor.add(Dropout(0.2))
	featureExtractor.add(Dense(1024, activation='relu', W_constraint=maxnorm(3),weights=vgg16.layers[14].get_weights()))
	#featureExtractor.add(Dropout(0.2))
	featureExtractor.add(Dense(512, activation='relu', W_constraint=maxnorm(3),weights=vgg16.layers[16].get_weights()))

	return featureExtractor
'''
These lines of code load in the VGG_16 model and then cut off the last few decision layers
'''

frogmodel=load_model(frogclassifier)
featureExtractor=CreateCNN(frogmodel)

#Define the LSTM model
model = Sequential()
model.add(GRU(128,return_sequences=False,stateful=False,input_shape=(mostfiles,512)))
model.add((Dense(100, activation='relu')))
model.add(Dropout(.5))
model.add((Dense(50, activation='relu')))
model.add(Dropout(.2))
model.add((Dense(25, activation='relu')))
model.add((Dense(1, activation='sigmoid')))
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])



#block of code to save the model
ts=time.time()
st=datetime.datetime.fromtimestamp(ts).strftime("%Y%m%d %H%M%S")
filename=modelsavename+st+'.h5'

if preprocessTrainingData:
	numberofTrains1=numberofSubTrains
	X_training1=np.zeros((numberofSubTrains,mostfiles,512))
	y_training1=np.zeros((numberofSubTrains,1))
	
	#If you are saving the data
	X_training1, y_training1 = BuildFiles(solutions1, numberofSubTrains, IMG_PX_SIZE, featureExtractor, X_training1, labeledTrainingData, y_training1)	
	np.save("preprocessVGG16\\X_trainingfrog.npz",X_training1)
	np.save("preprocessVGG16\\y_trainingfrog.npz",y_training1)
else:
	#If you are loading the data
	X_training=np.load("X_trainingfrog.npz.npy")
	y_training=np.load("y_trainingfrog.npz.npy")

	
for i in range(0,10):
	model.train_on_batch(X_training, y_training)

X_training2=X_training
X_training2=X_training2*np.random.uniform(.95,1.05,X_training2.shape)
y_training2=y_training


pbar = progressbar.ProgressBar(maxval=1).start()
for i in range(0,5):
	model.train_on_batch(X_training, y_training)
	model.train_on_batch(X_training2, y_training2)
	model.save(filename)
	pbar.update(i/5)
pbar.finish()

model.save(filename)

X_training=0
y_training=0
if preprocessTestingData:
	X_testing=np.zeros((numberofTests,mostfiles,512))
	y_testing=np.zeros((numberofTests,1))
	if labeledTestingData:
		X_testing, y_testing = BuildFiles(validationList, numberofTests, IMG_PX_SIZE, featureExtractor, X_testing, labeledTestingData, y_testing)	
		np.save("y_validationfrog.npz",y_testing)
	if not labeledTestingData:
		X_testing, directoryOrder = BuildFiles(validationList, numberofTests, IMG_PX_SIZE, featureExtractor, X_testing, labeledTestingData, y_testing)
		with open("validationOrder.txt","wb") as fp:
			pickle.dump(directoryOrder,fp)
			
	np.save("X_validationfrog.npz",X_testing)
else:
	X_testing=np.load("X_validationfrog.npz.npy")
	if labeledTestingData:
		y_testing=np.load("y_validationfrog.npz.npy")
	else:
		with open("validationOrder.txt",'rb') as fp:
			directoryOrder=pickle.load(fp)

X_testing=X_testing[:,0:mostfiles,]
predictions=model.predict(X_testing)

print(predictions)
results=pandas.DataFrame({'id':pandas.Series(directoryOrder),'cancer':pandas.Series(predictions.reshape(numberofTests))})

results.to_csv("cancerpredictions.csv",index=False)
