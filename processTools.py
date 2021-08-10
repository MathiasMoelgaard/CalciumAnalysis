import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import umap
import umap.plot
from sklearn import preprocessing

def probShift(df):
    return np.sum(np.array(df.iloc[:]), 0)/len(df.iloc[:])

#take in cell data and normalize and simplify by default
#parameters: SNR = Signal to Noise Ratio, value_cut is about what value is considered a valid spike.
def preprocessed_df(path, events = None, simplify = True, loc = False, SNR_cut = 6, value_cut = 200):
    
    #read temporal data
    df = pd.read_csv(path[0], skipinitialspace = True)
    
    #get cells to keep
    toAccept = df.columns[df.iloc[0] == "accepted"]
    
    #if event data, then use to narrow down accepted cells
    if events != None:
        #Get event data for accepted cells
        eventPdf = pd.read_csv(events[1], skipinitialspace = True)
        
        #Save full distribution for later comparisons
        eventPH = eventPdf
        
        #narrow down accepted cells based on event data
        eventPdf = eventPdf.loc[(eventPdf['Name'].isin(toAccept)) & (eventPdf['SNR'] > SNR_cut)]
        toAccept = eventPdf["Name"].tolist()

        #narrow down accepted cells again based on more event data
        eventdf = pd.read_csv(events[0], skipinitialspace = True)
        totals = eventdf.groupby(['Cell Name'])
        eventdf = eventdf.loc[(eventdf['Cell Name'].isin(toAccept)) & (eventdf['Value'] > value_cut)]
        toAccept = list(set(eventdf["Cell Name"].tolist()))
        
        #load cell size and location
        dfP = pd.read_csv(path[1], skipinitialspace = True)    
        jdf = pd.merge(dfP, eventPdf, on="Name")
        jdf = jdf.loc[(jdf['Name'].isin(toAccept)) & (jdf['Size'] < 20)]
        circle_size = jdf['SNR'] * jdf['Size'] / 2
        ax = jdf.plot.scatter(x='CentroidX',
                                      y='CentroidY',
                                      c='EventRate(Hz)', s=circle_size,
                                      colormap=plt.cm.magma)
        plt.figure()

        # Enumerate over the three columns of interest and make histograms that
        # show the accepted cells in one color, and rejected cells in another color.
        for k,col in enumerate(['SNR', 'EventRate(Hz)']):
            jdfN = pd.merge(dfP, eventPH, on="Name")
            jdfN = jdfN.loc[(~jdfN['Name'].isin(toAccept))]
            ax = plt.subplot(1, 3, k+1)
            n, bins, patches = plt.hist(jdf[col], bins=15, facecolor='#d69aff')
            plt.hist(jdfN[col], bins=bins, facecolor='k', alpha=0.5)
            plt.legend(['Accepted', 'Rejected'])
            plt.xlabel(col)
            plt.ylabel('Frequency')

        plt.show()
    
    #reread temporal data for final list of accepted cells
    df = pd.read_csv(path[0], skipinitialspace = True, usecols =[i for i in toAccept])
    #events = pd.read_csv(events[0], skipinitialspace = True)
    #drop classifying row
    df = df.iloc[1:]
    
    dfc = pd.DataFrame(0, columns=df.columns, index=df.index)
    
    #convert cell data to numeric
    df = df.apply(pd.to_numeric)
    
    # normalize all cells
    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis = 0)
    
    ###Alternative normalizing
    #df = np.array(df)
    #scaler = preprocessing.StandardScaler().fit(df)
    #df_scaled = scaler.transform(df)
    ###
    
    ###Many ways to preprocess data including a softmax approach.
    #Used method was choosen as it gave good results.
    ###
    
    #If choosed then convert to binary with a choosen limit for what is
    #considered an active cell vs deactive
    if simplify:
        events = pd.read_csv(events[0], skipinitialspace = True)
        events["Time (s)"] = (events["Time (s)"]*20).astype('int64')
        for i in events.index:
            try:
                index = events.loc[i, "Time (s)"]
                name = events.loc[i, "Cell Name"]
                if name in toAccept:
                    dfc.loc[index, name] = 1
            except:
                pass
        #convert cell data to numeric
        df = dfc.copy()
        #f = df.applymap(lambda x: 1 if x >= .6211 else 0) #golden ratio cutoff
        
    if loc == True:
        return df, jdf
    return df

def simplePrint(path, toAccept, events):
    
    #Get event data for accepted cells
    eventPdf = pd.read_csv(events[1], skipinitialspace = True)

    #narrow down accepted cells based on event data
    eventPdf = eventPdf.loc[(eventPdf['Name'].isin(toAccept))]
    toAccept = eventPdf["Name"].tolist()

    #narrow down accepted cells again based on more event data
    eventdf = pd.read_csv(events[0], skipinitialspace = True)
    totals = eventdf.groupby(['Cell Name'])
    eventdf = eventdf.loc[(eventdf['Cell Name'].isin(toAccept))]
    toAccept = list(set(eventdf["Cell Name"].tolist()))

    #load cell size and location
    dfP = pd.read_csv(path[1], skipinitialspace = True)    
    jdf = pd.merge(dfP, eventPdf, on="Name")
    jdf = jdf.loc[(jdf['Name'].isin(toAccept))]
    circle_size = jdf['SNR'] * jdf['Size'] / 2
    ax = jdf.plot.scatter(x='CentroidX',
                                  y='CentroidY',
                                  c='EventRate(Hz)', s=circle_size,
                                  colormap=plt.cm.viridis)
    plt.figure()

#to do
def heatMap(path, toAccept, events, clusters, df):
    
    fig, ax = plt.subplots(1, len(clusters) + 1)
    for i, cluster in enumerate(clusters):
        #Get event data for accepted cells
        eventPdf = pd.read_csv(events[1], skipinitialspace = True)

        #narrow down accepted cells based on event data
        eventPdf = eventPdf.loc[(eventPdf['Name'].isin(toAccept))]
        toAccept = eventPdf["Name"].tolist()

        #narrow down accepted cells again based on more event data
        eventdf = pd.read_csv(events[0], skipinitialspace = True)
        totals = eventdf.groupby(['Cell Name'])
        eventdf = eventdf.loc[(eventdf['Cell Name'].isin(toAccept))]
        toAccept = list(set(eventdf["Cell Name"].tolist()))

        #load cell size and location
        dfP = pd.read_csv(path[1], skipinitialspace = True)    
        jdf = pd.merge(dfP, eventPdf, on="Name")
        jdf = jdf.loc[(jdf['Name'].isin(toAccept))]
        jdf['EventRate(Hz)'] = (df.iloc[cluster].sum(axis=0)/df.sum(axis = 0)).to_numpy()
        circle_size = jdf['SNR'] * jdf['Size'] / 2
        ax[i] = jdf.plot.scatter(x='CentroidX',
                                      y='CentroidY',
                                      c='EventRate(Hz)', s=circle_size,
                                      colormap=plt.cm.viridis)
    plt.figure()
    
def AdvancedPrint(path, patterns, events, df):
    
    #Get event data for accepted cells
    eventPdf = pd.read_csv(events[1], skipinitialspace = True)

    #narrow down accepted cells based on event data
    eventPdf = eventPdf.loc[(eventPdf['Name'].isin(df.columns.tolist()))]
    toAccept = eventPdf["Name"].tolist()

    #narrow down accepted cells again based on more event data
    eventdf = pd.read_csv(events[0], skipinitialspace = True)
    totals = eventdf.groupby(['Cell Name'])
    eventdf = eventdf.loc[(eventdf['Cell Name'].isin(df.columns.tolist()))]
    toAccept = list(set(eventdf["Cell Name"].tolist()))

    #load cell size and location
    dfP = pd.read_csv(path[1], skipinitialspace = True)    
    jdf = pd.merge(dfP, eventPdf, on="Name")
    for k, v in patterns.items():
        try:
            v = sorted(v, key = lambda x: x[1][0])
        except:
            v = sorted(v, key = lambda x: x[1])
        fig, ax = plt.subplots(1, len(v) + 1)
        fig.suptitle('Temporal Clustering from {}'.format(k))
        temp = jdf.loc[(jdf['Name'].isin(k.split()))]
        circle_size = temp['SNR'] * temp['Size'] / 2
        temp.plot.scatter(x='CentroidX',
                                      y='CentroidY', s=circle_size,
                                      colormap=plt.cm.viridis, ax=ax[0])
        temp = jdf.loc[(jdf['Name'].isin(df.columns.tolist()))]
        circle_size = temp['SNR'] * temp['Size'] / 2
        temp.plot.scatter(x='CentroidX',
                                      y='CentroidY', s=circle_size,
                                      colormap=plt.cm.viridis, ax=ax[0], alpha = .1)
        ax[0].set_title('Parent Group')
        for i in range(len(v)):
            temp = jdf.loc[(jdf['Name'].isin(v[i][0]))]
            circle_size = temp['SNR'] * temp['Size'] / 2
            temp.plot.scatter(x='CentroidX',
                                          y='CentroidY', s=circle_size,
                                          colormap=plt.cm.viridis, ax=ax[i+1])
            
            temp = jdf.loc[(jdf['Name'].isin(df.columns.tolist()))]
            circle_size = temp['SNR'] * temp['Size'] / 2
            temp.plot.scatter(x='CentroidX',
                                          y='CentroidY', s=circle_size,
                                          colormap=plt.cm.viridis, ax=ax[i+1], alpha = .1)
            try:
                ax[i+1].set_title('{} ms'.format(v[i][1][0]*5))
            except:
                ax[i+1].set_title('{} ms'.format(v[i][1]*5))
        for i in range(len(v)+1):
            ax[i].set_ylim(0,300)
            ax[i].set_xlim(0,300)
        fig.set_size_inches(18.5, 10.5)
        plt.figure()
        
#Following is to get an idea of clusters
#Can be played with and wouldn't give any spatial information
def reduce_dimensions(df):
    # keep high dimensional structure in low dimension
    _umap = umap.UMAP(n_neighbors=3,
                     n_components=5,
                     min_dist = 0.65,
                     metric='jaccard').fit(df)
    umap_embeddings = _umap.transform(df)
    
    return _umap, umap_embeddings

def dbscan(embeddings, eps=3, min_samples=2):
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    return cluster.labels_

def show_prediction(df, clusters):
    # Prepare data
    bert_umap = umap.UMAP(n_neighbors=10,
                 n_components=2,
                 min_dist = 0.0,
                 metric='braycurtis').fit_transform(df)
    result = pd.DataFrame(bert_umap, columns=['x', 'y'])
    
    result['labels'] = clusters   # grab labels from clustering

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=100)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=100, cmap='viridis')

    plt.show()
    
def relu(x):
    result = []
    for i in x:
        if i >= 0:
            result.append(i)
        else:
            result.append(i/10)
    return result
