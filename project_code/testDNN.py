import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from sklearn import metrics

# CS551 Course work project Members:
#
# Keerthi Kumar S
# Mandeep Rathee
# Sandip Kishore
# IDS with deep learning approach Data set link: https://www.unb.ca/cic/datasets/ids-2017.html
# Github link: https://github.com/sandip-kishore/CS551-Deep-Learning-Apr-2019-Group11.git

# train and test directories
trainDir = 'trainSmall/testIDS.csv'
testDir = 'trainSmall/test_real_1KNeg.csv'
labelList = ['BENIGN','FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest','DoS GoldenEye','DoS Hulk',\
             'Heartbleed','Web Attack � XSS','Web Attack � Sql Injection','Web Attack � Brute Force','Infiltration','Bot',\
             'PortScan','DDoS']

# Function to load data and label mapping
def loadData(dirPath,labelList):
    dFrame = pd.read_csv(dirPath, skiprows=[0], header=None)
    # Assign classes to labels
    dFrame = dFrame.replace(labelList, range(len(labelList)))
    #dFrame = dFrame.replace(labelList, [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    dFrame = dFrame.sample(frac=1)
    data = dFrame.values
    data = np.float32(data)
    data = np.nan_to_num(data)
    features = data[:,range(data.shape[1]-1)]
    labels = data[:,-1]
    return (features,labels)

# Function to normalize the features
def normalizeData(featureList):
    featureList = np.log(abs(featureList)+1)
    fMax = np.max(featureList,axis=0)
    fMax[fMax==0] = 1
    fMin = np.min(featureList,axis=0)
    return ((featureList-fMin)/(fMax-fMin))


# Load the train features and labels
[features,labels] = loadData(trainDir,labelList)
featNorm = normalizeData(features)
dim = features.shape[1]

# Load the train features and labels
[testFeatures,testLabels] = loadData(testDir,labelList)
testFeatNorm = normalizeData(testFeatures)


# DNN model
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 78)                6162
# _________________________________________________________________
# dense_1 (Dense)              (None, 50)                3950
# _________________________________________________________________
# dense_2 (Dense)              (None, 25)                1275
# _________________________________________________________________
# dense_3 (Dense)              (None, 15)                390
# =================================================================
# Total params: 11,777
# Trainable params: 11,777
# Non-trainable params: 0
# Droputs can be uncommented if needed
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(dim,)),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(78,activation=tf.nn.relu),
    #keras.layers.Dropout(0.2),
    keras.layers.Dense(50,activation=tf.nn.relu),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(25,activation=tf.nn.relu),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(len(labelList),activation=tf.nn.softmax)
])

# Model training and validation
sgd = keras.optimizers.SGD(lr=0.1,momentum=0)
model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(featNorm,labels,epochs=500,validation_data=(testFeatNorm,testLabels))
test_loss, test_acc = model.evaluate(testFeatNorm, testLabels)
y_pred = model.predict(testFeatNorm)

# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

predOp = np.argmax(y_pred,axis=1)
conMat = tf.confusion_matrix(testLabels,predOp)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(conMat))
print('Test accuracy:', test_acc)
print(metrics.classification_report(testLabels,predOp,target_names=labelList))
print(model.summary())