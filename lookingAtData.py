import csv
import math
import operator
import scipy as sp
import sys


import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn import tree
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import svm

 

def GenX2data(X):
    Gfname = '/Users/katyghantous/Desktop/MachineLearningResearch/generation%s.csv'%(X);
    with open(Gfname, 'rU') as f:
        reader = csv.reader(f)
        Data = [['0']];
        for row in reader:
            Data.append(row)
    
    Data = np.delete(Data,(0),axis=0)

    sizeD = np.size(Data);  
    for i in range(sizeD):
        for j in range(np.size(Data[1])):
            Data[i][j] = (float) (Data[i][j]);



    return Data


def TestX2data(X):
    Gfname = '/Users/katyghantous/Desktop/MachineLearningResearch/test%s.csv'%(X);
    with open(Gfname, 'rU') as f:
        reader = csv.reader(f)
        Data = [['0']];
        for row in reader:
            Data.append(row)
                    
    Data = np.delete(Data,(0),axis=0)
                
    sizeD = np.size(Data);  
    for i in range(sizeD):
        for j in range(np.size(Data[1])):
            Data[i][j] = (float) (Data[i][j]);
                

    
    return Data

#def AppendGeneration(Data,X):
#    Gfname = 'generation%s.csv'%(X);
#    with open(Gfname, 'rU') as f:
#        reader = csv.reader(f)
#        for row in reader:
#            Data.append(row)
#    
#    
#    sizeD = np.size(Data)/np.size(Data[0]);  
#    for i in range(sizeD):
#        for j in range(np.size(Data[0])):
#            Data[i][j] = (float) (Data[i][j]);
#    
#    Data = np.delete(Data,(0),axis=0)
#    
#    return Data


def GetInput(Data,indecis):
    NumInp = np.size(indecis);
    if NumInp > (np.size(Data[0])-1):
        sys.exit('you assumed too many inputs! impossiaable!');
    
    for i in indecis:
        if i>(np.size(Data[0])):
            print i
            sys.exit('one of your assumed indecis for input is out of bount!! impossiaable!');
    
    Nx = np.size(Data);#/np.size(Data[0]); 
    Ny = (int) (NumInp);
    Inp = np.zeros((Nx,Ny))
    
    for i in range(Nx):
        k=0;
        for j in indecis:
            Inp[i][k] = Data[i][j];
            k=k+1;
    
    return Inp
    


def GetOutput(Data,indecis):
    NumOut = np.size(indecis);
    if NumOut > (np.size(Data[0])-1):
        sys.exit('you assumed too many outputs! impossiaable!');

    for i in indecis:
        if i>(np.size(Data[0])):
            print i
            sys.exit('one of your assumed indecis of output is out of bount!! impossiaable!');
        
    Nx = np.size(Data);#/np.size(Data[0]); 
    Ny = (int) (NumOut);
    Out = np.zeros((Nx,Ny))

    for i in range(Nx):
        k=0;
        for j in indecis:
            Out[i][k] = Data[i][j];
            k=k+1;
    
    return Out






######## multisim :) 

# lets look first at generation X
X = 9; 

# number of inputs 
indecisInp = range(9);
indexOut = [12];


Data = GenX2data(X);

Inp = GetInput(Data,indecisInp);
Out1 = GetOutput(Data,indexOut);



##
Datatest = TestX2data(X);
#
Inp_test = GetInput(Datatest,indecisInp);
Out1_test = GetOutput(Datatest,indexOut);

##
InpTr = zip(*Inp)
InptestTr = zip(*Inp_test)
##
OutTr = zip(*Out1)
OuttestTr = zip(*Out1_test)
##

for j in range(9):
    for i in range(9):
        line = plt.plot(np.sort(InpTr[j]),np.sort(InpTr[i])+100*i,'o-',np.sort(InptestTr[j]),np.sort(InptestTr[i])+100*i,'x-')
        r = random.random()
        b = random.random()
        c = random.random()
        plt.setp(line,color=[r,b,c])
    plt.show()



sys.exit('thats all folks!');

##

#plt.plot(OutTr)
#plt.plot(OuttestTr)
#plt.show()
##


#print 'computing svr...'

#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

#svr_poly = SVR(kernel='poly', C=1e3, degree=4)
#y_poly = svr_poly.fit(Inp, Out1).predict(Inp)

#y_poly = svr_rbf.fit(Inp, Out1).predict(Inp)

#plt.plot(Inp, y_poly)

#plt.scatter(Inp,Out1)


#plt.show()



####     desision tree works!

#clf = tree.DecisionTreeRegressor()
#clf.fit(Inp, Out1)

##clf = svm.SVR()
#print clf
#clf.fit(Inp, Out1)



clf = make_pipeline(PolynomialFeatures(4), Ridge())
clf.fit(Inp, Out1) 

#print clf.named_steps['polynomialfeature'].coef_

##### Testing

#Datatest = TestX2data(X);

#Inp_test = GetInput(Datatest,indecisInp);
#Out1_test = GetOutput(Datatest,indexOut);



predres = np.ones(np.size(Out1));
indx = 0;
for tstinp in Inp:
    predres[indx] = clf.predict(tstinp);
    indx = indx+1;


plt.plot(predres,'or')
plt.plot(Out1,'+r')
plt.show()

##### Testing 


predres_test = np.ones(np.size(Out1_test));
indx = 0;
for tstinp in Inp_test:
    predres_test[indx] = clf.predict(tstinp);
    indx = indx+1;


plt.plot(predres_test,'ob')
plt.plot(Out1_test,'+b')
for i in range(np.size(predres_test)):
    print 'actual %s vs predicted %s'%(Out1_test[i],predres_test[i])


plt.show()


