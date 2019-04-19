import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot

trainDir = 'trainSmall/testIDS.csv'
testDir = 'trainSmall/test_real_1KNeg.csv'
labelList = ['BENIGN','FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest','DoS GoldenEye','DoS Hulk',\
             'Heartbleed','Web Attack � XSS','Web Attack � Sql Injection','Web Attack � Brute Force','Infiltration','Bot',\
             'PortScan','DDoS']

def loadData(dirPath,labelList):
    dFrame = pd.read_csv(dirPath, skiprows=[0], header=None)
    # Assign classes to labels
    dFrame = dFrame.replace(labelList, range(len(labelList)))
    #dFrame = dFrame.replace(labelList, [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    #dFrame = dFrame.sample(frac=1)
    data = dFrame.values
    data = np.float32(data)
    data = np.nan_to_num(data)
    #data = data[~np.isnan(data).any(axis=1)]
    features = data[:,range(data.shape[1]-1)]
    labels = data[:,-1]
    return (features,labels)


def normalizeData(featureList):
    featureList = np.log(abs(featureList)+1)
    fMax = np.max(featureList,axis=0)
    fMax[fMax==0] = 1
    fMin = np.min(featureList,axis=0)
    return ((featureList-fMin)/(fMax-fMin))


# Load the train features and labels
[features,labels] = loadData(trainDir,labelList)
featNorm = normalizeData(features)
featNorm = featNorm[:,:,np.newaxis]
print(featNorm.shape)
dim = features.shape[1]

MINIBATCH_SIZE = 64
NUM_STEPS = len(featNorm)//MINIBATCH_SIZE

#np.savetxt('feat.csv',featNorm,delimiter=',')

# Load the train features and labels
[testFeatures,testLabels] = loadData(testDir,labelList)
testFeatNorm = normalizeData(testFeatures)
testFeatNorm = testFeatNorm[:,:,np.newaxis]


ipLayer = keras.layers.Input(shape=(dim,1,))
conv1 = keras.layers.Conv1D(filters=16,kernel_size=4,strides=2)(ipLayer)
maxPool = keras.layers.AveragePooling1D(pool_size=4)(conv1)
conv2 = keras.layers.Conv1D(filters=8,kernel_size=4,strides=1)(maxPool)
maxPool2 = keras.layers.AveragePooling1D(pool_size=4)(conv2)
flat = keras.layers.Flatten()(maxPool2)
hidden2 = keras.layers.Dense(72,activation='relu')(flat)
hidden3 = keras.layers.Dense(25,activation='relu')(hidden2)
opLayer = keras.layers.Dense(len(labelList),activation='softmax')(hidden3)
model = keras.models.Model(inputs=ipLayer,outputs=opLayer)
print(model.summary())

sgd = keras.optimizers.SGD(lr=0.1,momentum=0,decay=0.001)
model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(featNorm, labels,epochs=30,validation_data=(testFeatNorm,testLabels))
test_loss, test_acc = model.evaluate(testFeatNorm, testLabels)
y_pred = model.predict(testFeatNorm)

# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#correctPred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
predOp = np.argmax(y_pred,axis=1)
#accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
conMat = tf.confusion_matrix(testLabels,predOp)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(conMat))
print('Test accuracy:', test_acc)