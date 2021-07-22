import numpy as np
import math
import pandas as pd
import pyGMs as gm

#Loss function to try to optimize cutOffs
def loss(df, cellGroups, start, end):
    score = 0
    for cells in cellGroups:
        if len(cells) == 1:
            score *= .9
            continue
        probs = df.value_counts(cells, normalize=True)
        while isinstance(probs, pd.core.series.Series):
            try:
                probs = probs[1]
            except:
                break
        if isinstance(probs, pd.core.series.Series):
            continue
        for i in cells:
            for j in cells:
                if i == j:
                    continue
                y_probs = df.groupby(i).size().div(len(df))
                condProbY = df.groupby([i, j]).size().div(len(df)).div(y_probs, axis=0, level = i)
                try:
                    probs *= condProbY[1][1] + 1
                except:
                    probs *= .9
        score += probs*(1.61**len(cells))
        #print(score)
    return score*math.log(end-start, 10)/len(cellGroups)

#Recursive algorithm to get groups of cells that pass the cut-off on similarity
#Adjust cutOff to change group sizes
def getMaxGroup(group, x, graph, value, i, cutOff = .5):
    if value < 0 or i > 15:
        return 0
    vals = []
    valMax = -1
    for y in range(len(graph[x])):
        if y == x:
            vals.append(-1)
        else:
            vals.append(graph[x][y] - cutOff)
    if all(0 >= x for x in vals): #If no groupings can potentially improve then abandon current recursive call
        return 0
    for y in range(len(vals)):
        if vals[y] > 0 and y not in group:
            invalid = []
            val = 0
            for element in group:
                if graph[element][y] - cutOff < 0:
                    invalid.append(element)
            if len(invalid) < len(group)/2:
                valOp = 0
                for element in invalid:
                    valOp += graph[x][element] - cutOff
                if valOp < vals[y]: #if new cells score higher than all replacements then replace, can replace none and just add a cell
                    group.append(y)
                    for element in invalid:
                        group.remove(element)
                val = getMaxGroup(group, y, graph, vals[y]/len(group), i+1, cutOff = cutOff)
                
            if val < valMax and y in group: #If recursive call give sub-optimal solution then discard
                group.remove(y)
            elif y in group:
                valMax = val
    return valMax               

#Convert from numerical to cell names
def toCells(reducedGroups, cells):
    for group in reducedGroups:
        for count,x in enumerate(group):
            group[count] = cells[x]
    return reducedGroups

#Convert from cell to numberical
def toArrayDic(cells):
    myDic = {}
    for x, element in enumerate(cells):
        myDic[element] = x
    return myDic

class Graph:
    def __init__(self, df, hyperParams):
        #load graph class with cell data
        print(hyperParams)
        self.df = df.iloc[hyperParams[0]: hyperParams[1]]
        self.maxGroupCutOff = hyperParams[2]
        self.condXCutOff = hyperParams[3]
        self.condYCutOff = hyperParams[4]
        self.mutualCutOff = hyperParams[5]
        self.reduceCuttOff = hyperParams[6]
        
        #if start != None:
        #    self.df = self.df.iloc[start:]
        #if end != None:
        #    self.df = self.df.iloc[:end]
        self.toArrayDic(self.mmap())
        
    #Gather conditional probabilities based on current and next time slot    
    def create(self):
        #print("Start create")
        #Small value to ensure to division by zero
        xS = .00000001
        
        #Graphs with shape (#cells,#cells)
        self.graph = [[0 for y in range( len(self.df.columns))] for x in range( len(self.df.columns))]
        #self.graph = np.zeros([len(self.df.columns), len(self.df.columns)])
        #self.graphNext = np.zeros([len(self.df.columns), len(self.df.columns)])
        #self.graphNext = [[0 for y in range( len(self.df.columns))] for x in range( len(self.df.columns))]
        past = []
        for row in self.df.itertuples():
            #Get list of active cells at current timestamp
            #print(row[0])
            active = [x for x in range(0, len(row)-1) if row[x+1] == 1]
            #print(active)
            #ignore first iteration to allow for condition probabilities based on following values
            #This part can be alterned if the change between individual timestamps is minuscule 
            #if row[0] == 0:
            #    past = active
            #    continue
            #print(active)
            for x in active:
                for y in active:
                    #One more connection from x to y found
                    self.graph[x][y] +=1
            #for x in past:
            #    for y in active:
                    #One more connection from x past to y current found
            #        self.graphNext[x][y] +=1 
            #print(self.graph)
            #past = active

        #Set the range of all values to probability
        for x in range(len(self.df.columns)):
            val = self.graph[x][x] + xS
            #valNext = self.graphNext[x][x] + xS
            for y in range(len(self.df.columns)):
                self.graph[x][y] /= val
                #self.graphNext[x][y] /= valNext
        #print("End create")
    
    #If interested in probabilities over a range for how often cells appear
    def probShift(self, cells):
        if not isinstance(cells, list):
            return sum(np.array(self.df[cells].iloc[:]), 0)/len(self.df.iloc[:])
        return np.sum(np.array(self.df[cells].iloc[:]), 0)/len(self.df.iloc[:])
    
    def magEntropy(self, x):
        x_probs = self.df.groupby(x).size().div(len(self.df))
        try:
            return np.sum([-x_probs[0]*np.log2(x_probs[0]), -x_probs[1]*np.log2(x_probs[1])])
        except:
            return 0
    
    def magEntropyCom(self, x, y):
        probs = self.df.value_counts([x, y], normalize=True)
        result = 0
        y_probs = self.df.groupby(y).size().div(len(self.df))
        condProb = self.df.groupby([x, y]).size().div(len(self.df)).div(y_probs, axis=0, level=y)
        for i in range(int(len(probs)/2)):
            for j in range(len(probs[i])):
                try:
                    if j == 1 and i == 1:
                        result += probs[i][j]*np.log(condProb[i][j])
                    else:
                        result += probs[i][j]*np.log(condProb[i][j])
                except:
                    print(i, " : ", j, " : ", x, " : ", y)
        return -1*result
    
    def mutualInfo(self, x, y):
        return self.magEntropy(x) - self.magEntropyCom(x, y)
    
    #Returns list of columns names (cells)
    def mmap(self):
        return self.df.columns.tolist()

    #Find common elements between two groupings
    def countCluster(self, group1, group2):
        group1_as_set = set(group1)
        intersection = group1_as_set.intersection(group2)
        return len(intersection)

    #Group together cells based on 
    def grouping(self, graph):
        #grouping = graph.copy()
        grouping = []
        for x in range(len(graph)):
            memGroups = [x]
            group = getMaxGroup(memGroups, x, graph, 0, 0, cutOff = self.maxGroupCutOff) 
            #grouping[x] = [y for y in range(len(graph[x])) if graph[x][y] > .5]
            grouping.append(memGroups)
        return grouping
    
    #Group cells based on high mutual information
    def maxInfo(self):
        infomationRates = {}
        for x in self.mmap():
            for y in self.mmap():
                if x != y:
                    infomationRates[(x,y)] = self.mutualInfo(x,y)
                    
        #sorted list with highest mutual information processed first
        sorted_info = sorted(infomationRates.items(), key = lambda kv: kv[1], reverse = True)
        
        #Initialize weights for groups and groups
        weights = np.zeros(len(self.mmap()))
        grouping = [[i] for i in self.mmap()]
        
        #print(sorted_info)
        #Go through and update groups to get high mutual information
        for key, value in sorted_info:
            if value < .001:
                break
            y_probs = self.df.groupby(key[1]).size().div(len(self.df))
            x_probs = self.df.groupby(key[0]).size().div(len(self.df))
            condProbY = self.df.groupby([key[0], key[1]]).size().div(len(self.df)).div(y_probs, axis=0, level=key[1])
            condProbX = self.df.groupby([key[0], key[1]]).size().div(len(self.df)).div(x_probs, axis=0, level=key[0])
            try:
                if condProbX[1][1] > self.condXCutOff and condProbY[1][1] > self.condYCutOff:
                    cellId = self.toArray[key[0]]
                    cost = 0
                    for element in grouping[cellId]:
                        if element == key[1]:
                            cost -= 1
                        else:
                            cost += infomationRates[(key[1],element)]
                            cost += infomationRates[(element,key[1])]
                    cost /= len(grouping[cellId])*2

                    #Add cells if mutual information compared to current group is high enough
                    if cost > weights[cellId]*self.mutualCutOff:
                        weights[cellId] = cost
                        grouping[cellId].append(key[1])
            except:
                pass

        return grouping
                                                 
    #Group together groups with high level of similarity, adjust cutOff to change group sizes
    def reduceGroups(self, groups):
        i = 0
        while groups[i] == []:
            i +=1
        reducedGroups = [groups[i]]
        for x in range(i+1,len(groups)):
            unique = True
            for y in reducedGroups:
                #Get amount of shared cells
                count = self.countCluster(y, groups[x])
                
                #if count/(sum([len(groups[x]), len(y)+.000001])-count) >= cutOff:
                
                #If more cells than cutOff are shared then merge the two groups
                if count/(min([len(groups[x]), len(y)+.000001])) >= self.reduceCuttOff:
                    y.extend(groups[x])
                    #y.extend(groups[x])
                    unique = False
                if count == min([len(groups[x]), len(y)]):
                    unique = False
            if unique:
                reducedGroups.append(groups[x])
        result = []
        for y in reducedGroups:
            result.append(list(set(y)))
        return result
    
    def ChowLiu(self):
        X = [gm.Var(i,2) for i in range(len(self.df.columns.tolist()))]
        X = np.array_split(np.array(X), 16)
        factors = []
        
        for x in X: factors.append(gm.Factor(x,1.0))
            
        for i, factor in enumerate(factors):   
            for j in range(len(self.df)):
                xj = self.df.iloc[j:j+1, i*len(factors[i].vars) : (i+1)*len(factors[i].vars)]
                factor[xj] += 1.0  # count the various outcomes
            factor /= len(self.df)   
        
        model = gm.GraphModel(factors)
        phat = {}
        for i in model.X:         # estimate single-variable probabilities
            phat[i] = gm.Factor([i],1e-15)
            for xs in self.df.to_numpy(): phat[i][xs[i]] += 1.0
            phat[i] /= len(self.df)

        for i in model.X:         # estimate pairwise probabilities
            for j in model.X:
                if j<=i: continue
                phat[i,j] = gm.Factor([i,j],1e-15)
                for xs in self.df.to_numpy(): phat[i,j][xs[i],xs[j]] += 1.0
                phat[i,j] /= len(self.df)
                
        wts = np.zeros((108,108))
        for i in model.X:         # estimate pairwise probabilities
            for j in model.X:
                if j<=i: continue   # estimate (empirical) mutual information:
                wts[i,j] = (phat[i,j] * (phat[i,j]/phat[i]/phat[j]).log() ).sum()

        np.set_printoptions(precision=4, suppress=True)
        
        newFactors = []
        for i in model.X:
            val = 0
            for j in model.X:
                if wts[i,j] > val or wts[j,i] > val:
                    val = max(wts[i,j], wts[j,i])
            for j in model.X:
                if wts[i,j] == val or wts[j,i] == val:
                    break
            newFactors.append(gm.Factor([i,j], 1.0))
        est_model = gm.GraphModel(newFactors)
        gm.draw.drawMarkovGraph(est_model)
        
        groups = {}
        for x in est_model.factors:
            if x.vars[0] not in groups:
                groups[x.vars[0]] = []
            if x.vars[1] not in groups:
                groups[x.vars[1]] = []
            groups[x.vars[0]].append(x.vars[1])
            groups[x.vars[1]].append(x.vars[0])   
        
        myDic = groups
        def buildGroupHelper(current, depth):
            if current not in myDic or depth > 1:
                return []

            group = []
            for cell in myDic[current]:
                group.append(int(cell))
                group.extend(buildGroupHelper(cell, depth+1))

            myDic[current] = [] 
            return group

        def buildModelGroups(model): #dictionary
            explore = []
            groups = []
            for x in model.factors:
                group = []
                group.append(int(x.vars[0]))
                for cell in myDic[x.vars[0]]:
                    group.append(int(cell))
                    group.extend(buildGroupHelper(cell, 0))
                myDic[x.vars[0]] = []
                groups.append(group)
            return groups
        groups = buildModelGroups(est_model)
        
        group2 = []
        for group in groups:
            group2.append(list(set(group)))
            
        reducedGroups = self.reduceGroups(group2)
        cells = self.mmap()
        return toCells(reducedGroups, cells)
    
    #Convert from numerical to cell names
    def toCells(self, reducedGroups, cells):
        self.toCells = toCells(reducedGroups, cells)
        return self.toCells

    #Convert from cell to numberical
    def toArrayDic(self, cells):
        self.toArray = toArrayDic(cells)
