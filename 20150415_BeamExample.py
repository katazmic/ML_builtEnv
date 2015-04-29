import csv
import math
import operator
import scipy as sp

from sklearn import svm
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def calculateNormalizedWeight(additionalLoad,spanLength,shapeCollection):
    Fy = 50.0 #ksi
    omega = 2.67
    fAllow = Fy / omega #ksi
    E = 29000.0 #ksi
    
    acceptableDictionary = {}
    
    for shape in shapeCollection:
        uniformLoad = additionalLoad + shapeCollection[shape]["W"] 
        
        deflMax = spanLength * 12.0 / 360.0 #in
        Imin = (math.pow(12.0,3.0)/1000.0)*(5.0*uniformLoad*math.pow(spanLength,4.0))/(384.0*E*deflMax) #in^4
    
        Mmax = uniformLoad * math.pow(spanLength,2.0) / 8.0 #lb-ft
        Smin = (12.0/1000.0) * Mmax / fAllow #in^3
    
        Vmax = uniformLoad * spanLength / 2.0 #lbf
        Amin = Vmax / (1000.0 * fAllow) #in^2
        
        if ((shapeCollection[shape]["A"] > Amin) & (shapeCollection[shape]["I"] > Imin) & (shapeCollection[shape]["S"] > Smin)):
            acceptableDictionary[shape] = shapeCollection[shape]["W"]
            
    sortedAcceptable = sorted(acceptableDictionary.items(), key=operator.itemgetter(1))
    
    normalizedWeight = sortedAcceptable[0][1]
    print normalizedWeight
     
    return normalizedWeight

rootDirectory = "/Users/katyghantous/Desktop/MachineLearningResearch/"
shapeDatabaseFilename = "ShapeDatabase.csv"

shapeDictionary = {}

with open(rootDirectory+shapeDatabaseFilename, 'rU') as csvfile:
    shapeReader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(shapeReader, None)
    for row in shapeReader:
        rowDictionary = {"W":float(row[1]),"A":float(row[2]),"I":float(row[3]),"S":float(row[4])}
        shapeDictionary[row[0]] = rowDictionary

W = calculateNormalizedWeight(100.0,50.0,shapeDictionary)

X = np.arange(5., 1000., 100.)
Y = np.arange(5., 100., 10.)

Xm, Ym = np.meshgrid(X, Y)
Z = np.ones((np.size(X),np.size(Y)))


    
for i in range(0,np.size(X)):   
    for j in range(0,np.size(Y)):  
        Z[i][j] = calculateNormalizedWeight(X[i],Y[j],shapeDictionary)


print Z

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(Xm, Ym, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()




Ysd = np.arange(1.,150.,1.)
Zsd = np.ones(np.size(Ysd))
for j in range(0,np.size(Ysd)):  
    Zsd[j] = calculateNormalizedWeight(10.,Ysd[j],shapeDictionary);

plt.plot(Ysd,Zsd)
#plt.show()

Yin = [[Ysd[0]]];

for j in range(1,np.size(Ysd)):  
    Yin.append([Ysd[j]])


clf = make_pipeline(PolynomialFeatures(4), Ridge())

#clf = linear_model.Lasso(alpha = 0.1)
clf.fit(Yin, Zsd) 


pred = [clf.predict(Yin[0])];
for i in range(1,np.size(Ysd)):
    pred.append(clf.predict(Yin[i]))

plt.plot(Yin,pred,'-')



Rtestini = [1, 10, 32, 12, 23, 100, 80, 95, 120, 140, 45, 56]; #150.0*np.random.rand(1,10)


Rtest = [[Rtestini[0]]];

for j in range(1,np.size(Rtestini)):  
    Rtest.append([Rtestini[j]])


predR = [clf.predict(Rtest[0])];
for i in range(1,np.size(Rtest)):
    predR.append(clf.predict(Rtest[i]))


plt.plot(Rtest,predR,'o')

plt.show()









