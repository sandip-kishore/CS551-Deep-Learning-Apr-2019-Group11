import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot
from sklearn import metrics

trainDir = 'trainSmall/trainMon1kLess.csv'
testDir = 'trainSmall/test_real_1KNeg.csv'
labelList = ['BENIGN','FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest','DoS GoldenEye','DoS Hulk',\
             'Heartbleed','Web Attack � XSS','Web Attack � Sql Injection','Web Attack � Brute Force','Infiltration','Bot',\
             'PortScan','DDoS']

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



model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(dim,)),
    #keras.layers.Dropout(0.2),
    keras.layers.Dense(78,activation=tf.nn.relu),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(50,activation=tf.nn.relu),
    #keras.layers.Dropout(0.2),
    keras.layers.Dense(25,activation=tf.nn.relu),
    #keras.layers.Dropout(0.2),
    keras.layers.Dense(len(labelList),activation=tf.nn.softmax)
])
sgd = keras.optimizers.SGD(lr=0.1,momentum=0)
model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(featNorm,labels,epochs=50,validation_data=(testFeatNorm,testLabels))
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