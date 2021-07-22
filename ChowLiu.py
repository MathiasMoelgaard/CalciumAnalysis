#!/usr/bin/env python
# coding: utf-8

# Following code rely on the pyGMs library and tracks and reports dependency based on the ChowLiu algorithnm.

# In[ ]:


import pyGMs as gm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pylab

    
class ChowLiu:  
    def __init__(self, df):
        self.df = df
    
    def draw(self, model, loc, **kwargs):
        """Draw a Markov random field using networkx function calls
        Args:
          ``**kwargs``: remaining keyword arguments passed to networkx.draw()
        Example:
        >>> gm.drawMarkovGraph(model, labels={0:'0', ... } )    # keyword args passed to networkx.draw()
        """
        #Convert to cell notation
        toCell = {}
        
        presentCells = []
        for x in model.factors:
            presentCells.extend([x.vars[1], x.vars[0]])
    
        for i, cell in enumerate(self.df.columns.tolist()):
            if i in presentCells:
                toCell[i] = cell
        import networkx as nx
        import copy
        kwargs = copy.copy(kwargs)    # make a copy of the arguments dict for mutation
        G = nx.Graph()
        G.add_nodes_from( [v.label for v in model.X if v.states > 1] )  # only non-trivial vars
        for f in model.factors:
            for v1 in f.vars:
                for v2 in f.vars:
                    if (v1 != v2): G.add_edge(v1.label,v2.label)
        kwargs['var_labels'] = kwargs.get('var_labels',{n:n for n in G.nodes()})
        #kwargs['labels'] = kwargs.get('labels', kwargs.get('var_labels',{}) )
        kwargs.pop('var_labels',None)   # remove artificial "var_labels" entry)
        kwargs['edgecolors'] = kwargs.get('edgecolors','k')
        x = loc["CentroidX"].values.tolist()
        y = loc["CentroidY"].values.tolist()
        pos = {}
        for i in range(len(self.df.columns.tolist())):
            if i in presentCells:
                pos[i] = np.array([x[i], y[i]])
        pos
        plt.figure(figsize=(8,8))
        nx.draw(G, pos, labels = toCell, **kwargs)
        #nx.draw(G,**kwargs)
        return G
    
    def run(self, version, loc, threshold = .5):
        wts, model = self.ChowLiu() #Get and create factor model based on ChowLiu
        if version == "forest":
            model = self.largeForest(model, wts, threshold) ### for forest instead of Kruskals
        elif version == "kruskal":
            model = self.kruskals(model, wts)
        else:
            raise "Invalid version"
        groups = self.makeGroups(model) #Get groups from model
        
        self.draw(model, loc)
        return model, groups
    
    def kruskals(self, model, items):
        wts = items.copy()
        def findMax(model):
            maxCells = (0,0)
            for i in model.X:
                for j in model.X:
                    if wts[maxCells[0],maxCells[1]] < wts[i,j]:
                        maxCells = (i,j)
            return maxCells[0], maxCells[1]

        size = len(self.df.columns.tolist())
        graphWeights = np.zeros((size,size))

        explored = {}
        seen = []
        newFactors = []
        while len(newFactors) < size:
            i,j = findMax(model)
            if wts[i,j] == 0:
                break
            wts[i, j] = 0
            
            if j not in explored:
                explored[j] = [j]
            if i not in explored:
                explored[i] = [i]
                
            if j not in explored[i] and i not in explored[j] and j not in seen:
                explored[i].extend(explored[j])
                del explored[j]
                seen.append(j)
                newFactors.append(gm.Factor([i,j],1.0))
                
        est_model = gm.GraphModel(newFactors)
        #gm.draw.drawMarkovGraph(est_model)
        return est_model

    def largeForest(self, model, items, threshold = .5):
        size = len(self.df.columns.tolist())
        wts = items.copy()
        def findMax(model):
            maxCells = (0,0)
            for i in model.X:
                for j in model.X:
                    if wts[maxCells[0],maxCells[1]] < wts[i,j]:
                        maxCells = (i,j)
            return maxCells[0], maxCells[1]

        graphWeights = np.zeros((size,size))

        newFactors = []
        seen = {}
        
        while len(newFactors) < size:
            i,j = findMax(model)
            if wts[i,j] < threshold and len(newFactors) > size/1.61:
                break
            wts[i, j] = 0
            if i not in seen:
                seen[i] = 0
            seen[i] +=1
            if j not in seen:
                seen[j] = 0
            seen[j] +=1
            if seen[i] <= 10 and seen[j] <= 10:
                newFactors.append(gm.Factor([i,j],1.0))
        est_model = gm.GraphModel(newFactors)
        #gm.draw.drawMarkovGraph(est_model)
        return est_model

    def ChowLiu(self):
        #Initialize variables based on small bad model
        size = len(self.df.columns.tolist())
        X = [gm.Var(i,2) for i in range(size)]
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
                for xs in self.df.to_numpy(): 
                    phat[i,j][xs[i],xs[j]] += 1.0
                    if xs[i] == 1 and xs[j] == 1:
                        phat[i,j][xs[i],xs[j]] += 0
                phat[i,j] /= len(self.df)

        wts = np.zeros((size,size))
        for i in model.X:         # estimate pairwise probabilities
            for j in model.X:
                if j<=i: continue   # estimate (empirical) mutual information:
                wts[i,j] = (phat[i,j] * (phat[i,j]/phat[i]/phat[j]).log() ).sum()
        
        def softmax(xs):
            return np.exp(xs) / sum(np.exp(xs))
        
        wts = softmax(wts)
        
        wts = (wts - np.min(wts))/np.ptp(wts)
        return wts, model

    #Get groups of cells
    def makeGroups(self, model):
        def buildGroupHelper(current, depth, thresh = -1): #Don't deeply extend by default, change thresh to adjust
            if current not in myDic or depth > thresh:
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

        #Create short chains based on ChowLiu
        groups = {}
        for x in model.factors:
            if x.vars[0] not in groups:
                groups[x.vars[0]] = []
            if x.vars[1] not in groups:
                groups[x.vars[1]] = []
            groups[x.vars[0]].append(x.vars[1])
            groups[x.vars[1]].append(x.vars[0])   

        #Simple rename to use groups in recursive function
        myDic = groups

        #Groups based on ChowLiu
        groups = buildModelGroups(model)

        #Remove duplicates within groups
        group2 = []
        for group in groups:
            group2.append(list(set(group)))
        groups = group2

        #Remove duplicate groups and cells with no connections
        group2 = []
        for group in groups:
            if sorted(group) not in group2 and len(group) > 1:
                group2.append(sorted(group))
        groups = group2

        #Convert to cell notation
        toCell = {}
        for i, cell in enumerate(self.df.columns.tolist()):
            toCell[i] = cell

        group2 = []
        for group in groups:
            temp = []
            for cell in group:
                temp.append(toCell[cell])
            group2.append(temp)
        return group2 #Return Groups of cells

    

