from affinewarp import ShiftWarping, PiecewiseWarping
from statistics import mean
from processTools import dbscan, preprocessed_df, show_prediction, reduce_dimensions, relu, simplePrint
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import math
get_ipython().run_line_magic('matplotlib', 'inline')

def ATF(df, groups):
    result = []
    for group in groups:
        dfT = {}
        for e in group:
            for index in df.loc[(df[e] == 1)].index.tolist():
                if index not in dfT:
                    dfT[index] = 0
                dfT[index] += 1
                
        ts = []
        for key, val in dfT.items():
            if val >= len(group)*2/3:
                ts.append(key)
                
        final = []
        for e in ts:
            if not any(x in final for x in range(e-2, e)):
                final.append(e)
        result.append(final)
    return result

#AttainTimeFrames
def ATFSingle(df, group):
    result = []
    for e in group:
        dfT = df.loc[(df[e] == 1)]
        ts = dfT.index.tolist() #timeSlots
        final = []
        for e in ts:
            if not any(x in final for x in range(e-2, e)):
                final.append(e)
        result.append(final)
    return result

class wrappedData:
    def __init__(self, groups, paths, followBehavior = None):
        self.df = preprocessed_df(paths[0], paths[1], simplify = False)
        self.forAtf = preprocessed_df(paths[0], paths[1], simplify = True)
        if followBehavior != None:
            self.df.iloc[followBehavior]
            self.forAtf.iloc[followBehavior]
        #AttainTimeFrames
        self.atfFull = ATF(self.forAtf, groups)
        self.atfInd = []
        self.groups = groups
        for group in groups:
            self.atfInd.append(ATFSingle(self.forAtf, group))
    
    def getAft(self, groups):
        return ATF(self.forAtf, groups)
    
    def showWrapping(self, timedOn, cells, threshold):
        try:
            atf = ATFSingle(self.forAtf, timedOn)
        except:
            atf = ATF(self.forAtf, timedOn)
        print(atf)
        for cell in cells:
            self.getGroupSpikeSingle(cell, atf, timedOn, threshold = threshold)

    
    def findPatterns(self, baseTimes = None, groups = None):
        atfDic = {}
        spikeGroups = []
        if baseTimes == None and groups == None:
            baseTimes = self.atfInd
            groups = self.groups
        elif baseTimes != None and groups != None:
            assert(len(baseTimes) == len(groups))
        else:
            raise "timeSlots and groups to not match"
        for i, atf in enumerate(baseTimes):
            for j, timeSlots in enumerate(atf):
                for cell in self.df.columns.tolist():
                    spikes = self.getGroupSpikeSingle(cell, timeSlots, groups[i][j])
                    found = False
                    for k, spikeGroup in enumerate(spikeGroups):
                        count = 0
                        for spike in spikes:
                            for s in spikeGroup:
                                if spike in range(s-3, s+4):
                                    count +=1
                                    break
                        for spike in spikeGroup:
                            v = False
                            for s in spikes:
                                if spike in range(s-3, s+4):
                                    count +=1
                                    break
                        if count < (len(spikes) + len(spikeGroup))/1.61: #golden ratio            
                            pass
                        elif len(spikes) > 0:
                            found = True
                            try:
                                if groups[i][j] not in atfDic[k]:
                                    atfDic[k][groups[i][j]] = []
                                if cell not in atfDic[k][groups[i][j]]:
                                    atfDic[k][groups[i][j]].append(cell)
                            except:
                                g = ' '.join([str(elem) for elem in groups[i][j]])
                                if g not in atfDic[k]:
                                    atfDic[k][g] = []
                                if cell not in atfDic[k][g]:
                                    atfDic[k][g].append(cell)
                            break
                    if found == False and len(spikes) > 0:
                        spikeGroups.append(spikes)
                        try:
                            atfDic[len(spikeGroups)-1] = {groups[i][j]: [cell]}
                        except:
                            atfDic[len(spikeGroups)-1] = {' '.join([str(elem) for elem in groups[i][j]]): [cell]}
        patterns = {}
        for key, items in atfDic.items():
            if len(items) > 1 or any([len(val) > 1 for val in items.values()]):
                patterns[key] = items
        
        #Change format of dictionary to have an event as the key and a list of the groups follow that event as value
        newDic = {}
        for outerKey, innerDic in patterns.items():
            for innerKey, cellList in innerDic.items():
                if len(cellList) > 2:
                    if innerKey not in newDic:
                        newDic[innerKey] = []
                    newDic[innerKey].append((cellList, spikeGroups[int(outerKey)]))
 
        return newDic
        
    def findPatternsEx(self, baseTimes = None, groups = None):
        atfDic = {}
        spikeGroups = []
        if baseTimes == None and groups == None:
            baseTimes = self.atfInd
            groups = self.groups
        elif baseTimes != None and groups != None:
            assert(len(baseTimes) == len(groups))
        else:
            raise "timeSlots and groups to not match"
        for i, atf in enumerate(baseTimes):
            for j, timeSlots in enumerate(atf):
                for cell in self.df.columns.tolist():
                    spikes = self.getGroupSpikeSingle(cell, timeSlots, groups[i][j])
                    toRemove = []
                    for spike in spikes:
                        for k,s in enumerate(spikeGroups):
                            if spike in range(s-4, s+5):
                                if spike in range(s-1, s+2):
                                    toRemove.append(spike)
                                try:
                                    if groups[i][j] not in atfDic[k]:
                                        atfDic[k][groups[i][j]] = []
                                    if cell not in atfDic[k][groups[i][j]]:
                                        atfDic[k][groups[i][j]].append(cell)
                                except:
                                    g = ' '.join([str(elem) for elem in groups[i][j]])
                                    if g not in atfDic[k]:
                                        atfDic[k][g] = []
                                    if cell not in atfDic[k][g]:
                                        atfDic[k][g].append(cell)
                    toRemove = list(set(toRemove))
                    for seenSpike in toRemove:
                        spikes.remove(seenSpike)
                    
                    if len(spikes) > 0:
                        for spike in spikes:
                            spikeGroups.append(spike)
                            try:
                                atfDic[len(spikeGroups)-1] = {groups[i][j]: [cell]}
                            except:
                                atfDic[len(spikeGroups)-1] = {' '.join([str(elem) for elem in groups[i][j]]): [cell]}
        patterns = {}
        for key, items in atfDic.items():
            if len(items) > 1 or any([len(val) > 1 for val in items.values()]):
                patterns[key] = items
        
        #Change format of dictionary to have an event as the key and a list of the groups follow that event as value
        newDic = {}
        for outerKey, innerDic in patterns.items():
            for innerKey, cellList in innerDic.items():
                if len(cellList) > 2:
                    if innerKey not in newDic:
                        newDic[innerKey] = []
                    newDic[innerKey].append((cellList, spikeGroups[int(outerKey)]))
 
        return newDic
                                
                            
    def getGroupSpike(self, group, pioneerTS, baseSet, timeSlot = 110):
        models = []
        datas = []
        for e in group:
            data = [self.df.loc[x:x+timeSlot, [e]][e].tolist() for x in pioneerTS if x < len(self.df) - timeSlot]
            data = np.array(data)
            data = data.reshape(data.shape[0], data.shape[1], 1)
            datas.append(data)
            models.append(ShiftWarping(maxlag=.1, smoothness_reg_scale=10.))
            #models.append(PiecewiseWarping(n_knots=1, smoothness_reg_scale=10.))
            models[-1].fit(data, iterations=30)
            print(data.mean())
        plt.plot(sum([models[i].transform(datas[i])[:,:,0].mean(axis=0)[:] for i in range(len(models))]), color='k', label='warped average')
        plt.plot(sum([datas[i].mean(axis=0)[:] for i in range(len(datas))]), color='r', label='naive average')
        plt.ylabel('neural activity of {} from {}'.format(group, baseSet)), plt.xlabel('time (a.u.)'), plt.legend();
        plt.show()

    def getGroupSpikeSingle(self, e, pioneerTS, baseSet, timeSlot = 150, threshold = 10):
        if len(pioneerTS) < threshold:
            return []
        data = [self.df.loc[x:x+timeSlot, [e]][e].tolist() for x in pioneerTS if x < len(self.df) - timeSlot]
        data = np.array(data)
        data = data.reshape(data.shape[0], data.shape[1], 1)
        model = ShiftWarping(maxlag=.08, smoothness_reg_scale=10., warp_reg_scale=.02)
        #model = PiecewiseWarping(n_knots=1, smoothness_reg_scale=10.)
        model.fit(data, iterations=50)
        transformed = model.transform(data)[:,:,0].mean(axis=0)[:] #neglect immidiate result to focus on patterns following an event
        #std = np.std(transformed) #To make strong fluctuation curves less likely to consider something a spike
        std = stats.median_absolute_deviation(transformed) #to put less weight on outliers as we are looking for points that are outliers
        mean = transformed.mean()
        min_val = min(transformed)
        max_val = max(transformed)
        choosen = transformed
        if max(data.mean(axis=0)[:]) > max_val:
            choosen = data.mean(axis=0)[:]
        if max_val-min_val < .16:
            return []
        #print(std)
        #print(np.std(data.mean(axis=0)[30:]))
        count = 0
        timer = 0
        spikes = []
        spike_peak = 0
        baseLine = 100000
        for i in range(1,len(choosen)-20):
            if choosen[i-1] > choosen[i] and choosen[i+1] > choosen[i]:
                #n->p
                baseLine = choosen[i]
            elif choosen[i-1] < choosen[i] and choosen[i+1] < choosen[i] and choosen[i] > baseLine + std*3 and choosen[i] > .20 :
                #p->n
                if all(choosen[i + j] >= choosen[i]*math.exp(-j*.05/.20) for j in range(20)): #.5 seconds comes from 10*.05 which is the time between timeslots and .2 is a rough estimate of the mean lifetime in seconds
                    spikes.append(i)
                    count += 1
                
        """
        #Check if there is a spike large enough to be considered valid
        for i, point in enumerate(transformed[:len(transformed)-1]):
            if (point-mean)/(max_val-min_val) >= .8 and timer == 0:
                count = 1
                break
        
        #Track valid spikes if any
        if count > 0:
            count = 0
            for i, point in enumerate(transformed[:len(transformed)-1]):
                if point >= mean + std*2 and timer == 0:
                    count +=1
                    timer = 30
                    spike_peak = i
                elif timer > 0 and point > transformed[spike_peak]:
                    timer +=1
                    spike_peak = i
                elif timer > 0:
                    timer -= 1
                elif timer == 0 and transformed[spike_peak] > mean + std*2:
                    spikes.append(spike_peak)
                    spike_peak = i
        #print(max(transformed))
        #print("min: ", mean)
        #print("std: ", std)
        #print("count: ", count)
        """
        if len(spikes) > 0:#std/min_val > .02 and count > 0:
            #print("count: ", count)
            #print(spikes)
            plt.plot(transformed, color='k', label='warped average')
            plt.plot(data.mean(axis=0)[:] , color='r', label='naive average')
            plt.ylabel('neural activity of {} from {}'.format(e, baseSet)), plt.xlabel('time (a.u.)'), plt.legend();
            plt.show()
            return spikes
        return []

