from PIL import Image, ImageColor, ImageOps
import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
from skimage.exposure import histogram
from scipy import ndimage
import numpy as np
import cv2
import sys

#Remove potential background for recursive function
def rmBG(data):
    for x, row in enumerate(data):
        if row.all() == True:
            for y, element in enumerate(row):
                data[x][y] = False
    return data

#Recursive helper function to get cell, gathers all pixel locations where cells are found
#This algorithm will have decreased functionality on noisy data and only give rough estimates
#Works well on well preprocessed data
def helper(data, x, y, xSize, ySize, xMin, yMin, xMax, yMax):
    if x >= len(data) or y >= len(data[x]) or x < 0 or y < 0:
        return []
    #Define max size of cell if data is messy and noisy
    if data[x][y] == False:# or np.power(xSize - (xMax+xMin)/2, 2) + np.power(ySize - (yMax+yMin)/2, 2) > 600:
        return []
    data[x][y] = False
    obj = []
    obj.append((x,y))
    version = np.random.randint(9)
    if version == 0:
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax, max(yMax, ySize+1))) #d
    elif version == 1:
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax,max(yMax, ySize+1))) #d
    elif version == 2:
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax, max(yMax, ySize+1))) #d
    elif version == 3:
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax,  max(yMax, ySize+1))) #d
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
    elif version == 4:
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax, max(yMax, ySize+1))) #d
    elif version == 5:
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax, max(yMax, ySize+1))) #d
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
    elif version == 6:
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax, max(yMax, ySize+1))) #d
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
    elif version == 7:
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax, max(yMax, ySize+1))) #d
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
    elif version == 8:
        obj.extend(helper(data, x, y+1, xSize, ySize-1, xMin, min(yMin, ySize-1), xMax, yMax)) #c
        obj.extend(helper(data, x-1, y, xSize-1 , ySize, min(xMin, xSize-1), yMin, xMax, yMax)) #b
        obj.extend(helper(data, x+1, y, xSize+1, ySize, xMin, yMin, max(xMax, xSize+1), yMax)) #a
        obj.extend(helper(data, x, y-1, xSize, ySize+1, xMin, yMin, xMax, max(yMax, ySize+1))) #d
    return obj


###Ignore
def findMid(data, x, y, sizeX, sizeY, size):
    while (sizeX < size or sizeY < size) and x+sizeX+1 < len(data) and y+sizeY+1 < len(data[x]):
        if data[x+sizeX+1][y+sizeY+1] == False:
            sizeX +=1
            sizeY +=1
        elif data[x+sizeX+1][y+sizeY] == False:
            sizeX +=1
        elif data[x+sizeX][y+sizeY+1] == False:
            sizeY +=1
        else:
            break
        data[x+sizeX][y+sizeY] = True
    return sizeX/2, sizeY/2

###Ignore
def makeCell(x, y, xMid, yMid):
    obj = []
    for xx in range(int(-xMid), int(xMid)):
        for yy in range(int(-yMid), int(yMid)):
            if np.power(xx, 2) + np.power(yy, 2) < np.power(xMid, 2) + np.power(yMid, 2):
                obj.append((int(x+xMid+xx),int(y+yMid+yy)))
    return obj

#Goes through pixels in image looking for cells and when cell found
#expand recursively from there to get the whole cell while removing it from
#the original image. Sort the cell found for better visual.
def findObjects(data, size):
    objs = []
    for x,row in enumerate(data):
        for y,val in enumerate(row):
            if val == True:
                obj = helper(data, x, y, 0, 0, 0, 0, 0, 0)
                #xMid, yMid = findMid(data, x,y, 0, 0, size)
                #obj = makeCell(x, y, xMid, yMid)
                if len(obj) > size:
                    obj = sorted(obj, key=getKey)
                    objs.append(obj)
    return objs

#Put objects into image array
def toImageData(dataTuple, dataImage):
    change = 0.0
    #print(dataTuple)
    for obj in dataTuple:
        change += .2
        for x,y in obj:
            dataImage[x][y] = 1#change%1
    return dataImage

#Put object into image array
def toImageDataCell(obj, dataImage):
    for x,y in obj:
        dataImage[x][y] = 1.0
    return dataImage

def getKey(item):
     return item[0]

#Convert from rbg to grayscale
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class imageM:
    def __init__(self, path, maxheight = 1500, maxwidth = 1500):
        image = Image.open(path)
        image = ImageOps.grayscale(image)
        width, height = image.size 
        scale = min(maxwidth/width, maxheight/height)
        width *= scale
        height *= scale
        image = image.resize((int(width),int(height)), Image.ANTIALIAS)
        image.show()
        self.data = np.asarray(image)
        self.processed = False
        self.objOrder = [""]
        #["C103", "C117", "C102", "C131", "C003", "C114", "C125", "C022", "C064", "C096", "C077", "C110", "C047", "C106"]#[" C080", " C025", " C041", " C095", " C071", " C046", " C098", " C061", " C022", " C068", " C076", " C074", " C036", " C067", " C037", " C003", " C057", " C015", " C008", " C005", " C053", " C023", " C056", " C051", " C014", " C017", " C028", " C062", " C024", " C031", " C064", " C009", " C063", " C079", " C030", " C059", " C052", " C026", " C020", " C001", " C004", " C007", " C072", " C033", " C123", " C045", " C043", " C018", " C042", " C021", " C029", " C040", " C010", " C027", " C050", " C013", " C038", " C054", " C090"]
    
    #Gives cells with their spartial information
    def preprocess(self):
        self.processed = True
        
        #Get threshold to minimize noise
        val = filters.threshold_otsu(self.data)
        #Increase threshold for better cell outline with trace
        #Note: increasing thresh may not be efficient on raw images
        dataDNoised = self.data > 240#val+90
        DNoised = Image.fromarray(dataDNoised)
        filledCells = ndimage.binary_fill_holes(dataDNoised)
        fillImage = Image.fromarray(filledCells)
        #Print image after some image cleaning
        fillImage.show()
        fill = np.array(filledCells)
        #This just makes sure the background is always get to the same
        fill = rmBG(filledCells)
        self.data = filledCells.copy()
        #Finds cells through recursive algorithm to get spatial information for each cell
        #Takes in preprocessed image and size of which cells below that are ignored in case of noise remaining
        self.obj = findObjects(filledCells, 200)
    
    #Visualize image with data as well as return array version
    def imageData(self):
        if self.processed == False:
            raise "Not preprocessed"
        empty = np.zeros(self.data.shape[0]*self.data.shape[1]).reshape(self.data.shape[0], self.data.shape[1])
        #empty = empty.astype(bool)
        imageData = toImageData(self.obj, empty)
        imageData = (imageData*255).astype(np.uint8)
        image = Image.fromarray(imageData)
        image.show()
        return imageData
       
        
    #Uses spartial data from objOrder and groupings data from myGraph to visualize different groups
    def displayGroups(self, groups, myGraph):
        cellToArray = myGraph.toArrayDic(self.objOrder)
        sharedCells = {}
        groupings = np.zeros(self.data.shape[0]*self.data.shape[1]*3).reshape(self.data.shape[0], self.data.shape[1], 3)
        groupings = (groupings).astype(np.uint8)
        for grouping in groups:
            group = []
            for x, obj in enumerate(grouping):
                try:
                    current = str(self.obj[cellToArray[obj]])
                    if current not in sharedCells:
                        sharedCells[current] = 0
                    sharedCells[current] +=1
                    objArea = []
                    for x, y in zip(self.obj[cellToArray[obj]][0::2], self.obj[cellToArray[obj]][1::2]):
                        objArea.extend([x,y])
                        
                    ###
                    #Allow a cell to be shared with up to 4 groupings
                    #Can be improved and implemented in different ways but this gives a nice visual
                    #as long as the about of groups you are looking at at once don't exceed cells being
                    #shared more than 4 times. The more unique the groupings, for the efficient.
                    if sharedCells[current] == 2:
                        objArea = objArea[:int(len(objArea)/2)]
                    if sharedCells[current] == 3:
                        objArea = objArea[int(len(objArea)/4):int(len(objArea)/4) + int(len(objArea)/2)]
                    if sharedCells[current] == 4:
                        objArea = objArea[int(len(objArea)/4):int(len(objArea)/2)]
                    group.extend(objArea)
                except:
                    pass
            #Takes data from array to image data preserving previous objects added to data and modifying
            #to adjust for shared cells
            empty = np.zeros(self.data.shape[0]*self.data.shape[1]).reshape(self.data.shape[0], self.data.shape[1])
            imageData2 = toImageDataCell(group, empty)
            imageData2 = (imageData2).astype(np.uint8)
            color = np.random.randint(255, size=3)
            imageData2 = cv2.cvtColor(imageData2,cv2.COLOR_GRAY2RGB)
            imageData2 = (imageData2*color).astype(np.uint8)
            update = imageData2 > 0
            groupings[update] = imageData2[update]
        image5 = Image.fromarray(groupings)
        image5.show()

