# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
# load data
(X_train, y_train), (X_train, y_train) = cifar10.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(toimage(X_train[i]))
# show the plot

# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import scipy
from skimage import color
from skimage import io

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

IMG_PX_SIZE=150

def rgb2grey(X):
	img=X[0]*299./1000+X[1]*587./1000+X[2]*114./1000
	return(img)
	
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
def changeData(X_train,y_train,firstclass,secondclass):
	X_train = X_train.astype('float32')
	X_train = X_train / 255.0
	is0=y_train==firstclass
	is1=y_train==secondclass

	y_train0=y_train[is0]
	y_train1=y_train[is1]

	class0=[0 for y in y_train0]
	class1=[1 for y in y_train1]

	X_train0=X_train[numpy.where(is0)[0]]
	X_train1=X_train[numpy.where(is1)[0]]

	y_training=numpy.concatenate((class0,class1),axis=0)
	y_training =np_utils.to_categorical(y_training)
	y_trian =np_utils.to_categorical(y_train)
	X_training=numpy.concatenate((X_train0,X_train1),axis=0)
	num_classes=y_training.shape[1]

	trainingshape=(X_training.shape[0],1,IMG_PX_SIZE,IMG_PX_SIZE)
	X_trainingdata=numpy.zeros(trainingshape)
	for x,i in zip(X_training,range(0,X_trainingdata.shape[0])):
		x=rgb2grey(x)
		X_trainingdata[i][0,:,:]=scipy.misc.imresize(numpy.array(x),(IMG_PX_SIZE,IMG_PX_SIZE))/255.0
	
	return(X_trainingdata,y_training,num_classes)

[X_trainingdata,y_training,num_classes]=changeData(X_train,y_train,6,9)
[X_testingdata,y_testing,num_classes]=changeData(X_test,y_test,6,9)

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(1, IMG_PX_SIZE, IMG_PX_SIZE), activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))


# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
# Fit the model
			
model.fit(X_trainingdata, y_training, validation_data=(X_testingdata, y_testing), nb_epoch=epochs, batch_size=32)

model.save("CNNfromgrayscalecifar10data100by100.h5")

# Final evaluation of the model
scores = model.evaluate(X_testingdata, y_testing, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))