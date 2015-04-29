import csv
import math
import operator
import scipy as sp
import sys


from sklearn.svm import SVR
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


def GenX2data(X):
    Gfname = 'generation%s.csv'%(X);
    with open(Gfname, 'rU') as f:
        reader = csv.reader(f)
        Data = [['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']];
        for row in reader:
            Data.append(row)
    

    sizeD = np.size(Data)/np.size(Data[0]);  
    for i in range(sizeD):
        for j in range(np.size(Data[0])):
            Data[i][j] = (float) (Data[i][j]);
                
    Data = np.delete(Data,(0),axis=0)

    return Data


def TestX2data(X):
    Gfname = 'test%s.csv'%(X);
    with open(Gfname, 'rU') as f:
        reader = csv.reader(f)
        Data = [['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']];
        for row in reader:
            Data.append(row)
                    

    sizeD = np.size(Data)/np.size(Data[0]);  
    for i in range(sizeD):
        for j in range(np.size(Data[0])):
            Data[i][j] = (float) (Data[i][j]);
                
    Data = np.delete(Data,(0),axis=0)
    
    return Data


def GetInput(Data,indecis):
    NumInp = np.size(indecis);
    if NumInp > (np.size(Data[0])-1):
        sys.exit('you assumed too many inputs! impossiaable!');
    
    for i in indecis:
        if i>(np.size(Data[0])):
            print i
            sys.exit('one of your assumed indecis for input is out of bount!! impossiaable!');
    
    Nx = np.size(Data)/np.size(Data[0]); 
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
            sys.exit('one of yout assumed indecis of output is out of bount!! impossiaable!');
        
    Nx = np.size(Data)/np.size(Data[0]); 
    Ny = (int) (NumOut);
    Out = np.zeros((Nx,Ny))

    for i in range(Nx):
        k=0;
        for j in indecis:
            Out[i][k] = Data[i][j];
            k=k+1;
    
    return Out






######## multisim :) 

# lets look first at generation 0
X = 0; 

# number of inputs 
indecisInp = range(10);
indexOut = [10];


Data = GenX2data(X);

Inp = GetInput(Data,indecisInp);
Out1 = GetOutput(Data,indexOut);

#print 'computing svr...'

#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

#svr_poly = SVR(kernel='poly', C=1e3, degree=4)
#y_poly = svr_poly.fit(Inp, Out1).predict(Inp)

#y_poly = svr_rbf.fit(Inp, Out1).predict(Inp)

#plt.plot(Inp, y_poly)

#plt.scatter(Inp,Out1)


#plt.show()





clf = make_pipeline(PolynomialFeatures(4), Ridge())

#clf = linear_model.Lasso(alpha = 0.1)
clf.fit(Inp, Out1) 



##### Testing
predres = np.ones(np.size(Out1));
indx = 0;
for tstinp in Inp:
    predres[indx] = clf.predict(tstinp);
    indx = indx+1;


plt.plot(predres,'r')
plt.plot(Out1,'+r')
plt.show()

##### Testing 


Datatest = TestX2data(X);

Inp_test = GetInput(Datatest,indecisInp);
Out1_test = GetOutput(Datatest,indexOut);
predres_test = np.ones(np.size(Out1_test));
indx = 0;
for tstinp in Inp_test:
    predres_test[indx] = clf.predict(tstinp);
    indx = indx+1;


plt.plot(predres_test,'b')
plt.plot(Out1_test,'+b')
for i in range(np.size(predres_test)):
    print 'actual %s vs predicted %s'%(Out1_test[i],predres_test[i])


plt.show()

#tryX = 10;
#pred = [clf.predict(Inp[tryX])];

#print 'predicted value is %s' %(pred);
#print 'actual value is %s' %(Out1[tryX]);


#
#    
#Ysd = np.arange(1.,150.,1.)
#Zsd = np.ones(np.size(Ysd))
#for j in range(0,np.size(Ysd)):  
#    Zsd[j] = calculateNormalizedWeight(10.,Ysd[j],shapeDictionary);
#
#Yin = [[Ysd[0],Ysd[0]]];
#
#for j in range(1,np.size(Ysd)):  
#    Yin.append([Ysd[j],Ysd[j]])
#
#
#clf = make_pipeline(PolynomialFeatures(4), Ridge())
#
##clf = linear_model.Lasso(alpha = 0.1)
#clf.fit(Yin, Zsd) 
#
#
#pred = [clf.predict(Yin[0])];
#for i in range(1,np.size(Ysd)):
#    pred.append(clf.predict(Yin[i]))
#
#plt.plot(Yin,pred,'-')
#
#
#
#Rtestini = [1, 10, 32, 12, 23, 100, 80, 95, 120, 140, 45, 56]; #150.0*np.random.rand(1,10)
#
#
#Rtest = [[Rtestini[0],Rtestini[0]]];
#
#for j in range(1,np.size(Rtestini)):  
#    Rtest.append([Rtestini[j],Rtestini[j]])
#
#
#predR = [clf.predict(Rtest[0])];
#for i in range(1,np.size(Rtest)):
#    predR.append(clf.predict(Rtest[i]))
#
#
#plt.plot(Rtest,predR,'o')
#
#plt.show()
#
#
#
#






