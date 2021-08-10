#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from processTools import dbscan, preprocessed_df, show_prediction, reduce_dimensions, relu, simplePrint
import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import numpy as np, pandas as pd
from tensorflow import losses, optimizers
from tensorflow.keras import Input, Model, models, layers

#Taken from https://blog.finxter.com/calculating-the-angle-clockwise-between-2-points/
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def bestSquare(cols):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    x = 1
    y = len(cols)
    while y > x: #Get the closest to a perfect square
        yOld = y
        xOld = x
        for i in range(2,y+1):
            if y%i == 0:
                y = int(y/i)
                x = int(x*i)
                break
        if abs(xOld-yOld) < abs(x-y):
            x = max(xOld, yOld)
            y = min(xOld, yOld)
            break
    return x,y

def getNextLayer(curLayer, x):
    nextSet = set()
    for i in curLayer:
        for j in range(1,x+1):
            if (i[0]+j, i[1]+1) not in curLayer:
                nextSet.add((i[0]+j, i[1]+1))
            if (i[0], i[1]+1) not in curLayer:
                nextSet.add((i[0], i[1]+1))
            if (i[0]+j, i[1]) not in curLayer:
                nextSet.add((i[0]+j, i[1]))
    curLayer = list(nextSet)
    curLayer = sorted(curLayer, key = lambda x: x[1], reverse = True)
    return sorted(curLayer, key = lambda x: x[0])
        
def changeColOrder(df, loc):
    cols = np.array(df.columns)
    x,y = bestSquare(cols)
    dist = []
    distMatrix = np.ones((len(df.columns), len(df.columns)))*10000
    for i in range(len(cols)):
        e1 = loc[loc['Name'] == cols[i]][["CentroidX", "CentroidY"]].values
        dist.append((np.sqrt(e1.dot(e1.T))[0], angle_between(e1[0], [0,0]), i))
        for j in range(len(cols)):
            if i != j:
                e2 = loc[loc['Name'] == cols[j]][["CentroidX", "CentroidY"]].values
                dif = e1-e2
                distMatrix[i,j] = np.sqrt(dif.dot(dif.T))

    dist = sorted(dist, key = lambda x: x[0][0])
    
    newCols = np.ones((x, y))*-1
    curLayer = [(0,0)]
    for i in range(1, y+1):
        elements = i*int(x/y*i) - ((i-1) * int(x/y*(i-1)))
        curList = []
        while elements > 0:
            curList.append(dist.pop(0))
            elements -=1
        curList = sorted(curList, key = lambda x: x[1]*x[0][0])
        for j in range(len(curLayer)):
            newCols[curLayer[j]] = curList[j][2]
        curLayer = getNextLayer(curLayer, int(x/y*(i+1)) - int(x/y*(i)))
    newCols = newCols.flatten()    
    
    return [cols[int(x)] for x in newCols]

def createTargets(annot, times = 20):
    result = []
    result.extend([0]*(annot[0])*times)
    for i in range(len(annot)-1):
        if i % 2 == 0:
            result.extend([1]*(annot[i+1] - annot[i])*times)
        else:
            result.extend([0]*(annot[i+1] - annot[i])*times)
    #result.extend([0]*(len(df) - annot[-1]*times))
    return result

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    x = 1
    y = array.shape[-1]
    while y > x: #Get the closest to a perfect square
        yOld = y
        xOld = x
        for i in range(2,y+1):
            if y%i == 0:
                y = int(y/i)
                x = int(x*i)
                break
        if abs(xOld-yOld) < abs(x-y):
            x = max(xOld, yOld)
            y = min(xOld, yOld)
            break
    print(array.shape)
    try:
        array = np.reshape(array, (array.shape[0], array.shape[1], x, y, 1))
    except:
        array = np.reshape(array, (array.shape[0], x, y, 1))
    return array
    
class LSTM_CNN:
    def __init__(self, annots, df, baseCase = True, timeStamps = 20):
        self.timeStamps = timeStamps
        if baseCase:
            self.annots = createTargets(annots)
            self.annots.extend([1]*10)
            self.annots = np.array(self.annots).reshape(-1,1)
        else:
            self.annots = annots
        self.df = df
    
    def rearrangeCols(self, loc):
        newCols = changeColOrder(self.df, loc)
        self.df = self.df[newCols]
        
    def dataPrep(self, dfE = None, downSample = 1, binary = False, random_state = 1):#Shape data into image format
        features =  np.array(self.df)
        x = []
        y = []

        if isinstance(dfE, pd.DataFrame):
            if binary:
                features =  np.array(dfE)
            indexes = dfE[dfE.sum(axis=1) != 0].index
            indexes = indexes[indexes < len(self.df)-self.timeStamps] -1
            for i in range(len(features)-self.timeStamps-2):
                if i in indexes:
                    x.append(features[i:i+self.timeStamps:downSample])
                    y.append(self.annots[i+self.timeStamps-1])
        else:
            for i in range(len(features)-self.timeStamps-2):
                x.append(features[i:i+self.timeStamps:downSample])
                y.append(self.annots[i+self.timeStamps-1])
        x = np.array(x)
        y = np.array(y)

        x = preprocess(x)
        
        from sklearn.model_selection import train_test_split   
        self.X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = random_state)
        return self.X_train, X_test, y_train, y_test
        
    def build(self):
        try: 
            t = self.X_train
        except:
            raise("Data hasn't been processed for building")
            
        inputs = Input(shape=(self.X_train.shape[1],self.X_train.shape[2], self.X_train.shape[3], self.X_train.shape[4]))
        #lstm = lambda x, channels: layers.LSTM(channels)(x)
        cnn_feat = layers.TimeDistributed( layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))(inputs)
        cnn_feat = layers.TimeDistributed( layers.Conv2D(filters = 128, kernel_size = 3, strides = 2, activation = 'relu'))(cnn_feat)
        cnn_feat = layers.TimeDistributed( layers.Dropout(0.5))(cnn_feat)
        cnn_feat = layers.TimeDistributed( layers.MaxPooling2D(2))(cnn_feat)
        cnn_feat = layers.TimeDistributed( layers.Flatten())(cnn_feat)
        l1 = layers.LSTM(32)(cnn_feat)
        l2 = layers.Dropout(0.5)(l1)
        l3 = layers.Dense(32)(l2)
        l4 = layers.Dense(self.annots.shape[-1], activation = 'softmax')(l3)
        model = Model(inputs = inputs, outputs = l4)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=2e-4),
            loss="binary_crossentropy", 
            metrics=['accuracy'])
        return model
        
        
    def cellAccurate(self, models, X_test, dfE, delay = 0, downSample = 1):
        #predict
        activitySize = {}
        sizeCount = {}
        for i, model in enumerate(models):
            X_train, X_test, y_train, y_test = self.dataPrep(dfE, downSample = 1, random_state = i)
            y_pred = model.predict(X_test)
            X_train, X_test, y_train, y_test = self.dataPrep(dfE, downSample = downSample, binary = True, random_state = i)
            for i, data_pt in enumerate(X_test):
                if i + delay >= len(X_test) or i + delay < 0:
                    continue
                cellsActive = round(data_pt.sum()/data_pt.shape[0])
                if cellsActive not in activitySize:
                    activitySize[cellsActive] = 0
                    sizeCount[cellsActive] = 0
                if len(y_test) == 1:
                    activitySize[cellsActive] += round(y_pred[i + delay][0]) == y_test[i + delay][0]
                else:
                    activitySize[cellsActive] += np.array_equal(y_pred[i + delay] == y_pred[i + delay].max(), y_test[i + delay])
                sizeCount[cellsActive] +=1

        #plot
        plot_data = []
        for key in sorted(activitySize.keys()):
            if sizeCount[key] > 200:
                plot_data.append(activitySize[key]/sizeCount[key]*100)
            
        import matplotlib.pyplot as plt
        plt.plot(sorted(activitySize.keys())[:len(plot_data)], plot_data)
        print(sizeCount)         

