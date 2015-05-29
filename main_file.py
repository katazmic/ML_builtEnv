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


from ml_funcs import *




#### setting up the data to be analyzed

Data_M = GetData();
Data_shuf = shuffle(Data_M,512);  # to shuffle around the data before taking a random sample

indecisInp = range(9);
indexOut = [9];
npoly = 4;


#rmsArr =evaluate_fit_poly(Data_M,indecisInp,indexOut,npoly);



#print rmsArr
#sys.exit('thats it for now')

##############################

Nperc = 7;
NprTrls = 1;


meth = 0; # 0 for tree 1 for poly
error = np.zeros((Nperc,NprTrls));
mean = np.zeros(Nperc);
std = np.zeros(Nperc);
Ntrain = np.zeros(Nperc);

  
percentageTst0 = 2.;

sizeTst = (int) (np.size(Data_shuf)*percentageTst0/100.);
print sizeTst
sizeTrn = (int) ((np.size(Data_shuf)-2*sizeTst)/Nperc);
print sizeTrn
percentageTrn0 = ((float) (sizeTrn))/np.size(Data_shuf)*100.;


Outv = np.zeros((Nperc,sizeTst))
Outp = np.zeros((Nperc,sizeTst))




for i in range(Nperc):
    mn = 0;
    if i==0:
        Data_trn0, Data_tst0, Data_rem0 = random_TrnTst(Data_shuf,percentageTrn0,percentageTst0);
    for j in range(NprTrls):
        if NprTrls !=1:
            Data_tst0 = get_randTest(Data_rem0,sizeTst)
        #Data_tst0 = get_randTest(Data_trn0,sizeTst)
        if meth ==0:
            Out_trn,Out_predtrn, Out_tst,  Out_predtst, rmsEr =find_pred_tree(Data_trn0,Data_tst0,indecisInp,indexOut);
        if meth ==1:
            Out_trn,Out_predtrn, Out_tst,  Out_predtst, rmsEr =find_pred_poly(Data_trn0,Data_tst0,indecisInp,indexOut,npoly);
        error[i][j] = rmsEr;
        mn = mn + rmsEr;
    
    mean[i] = mn/NprTrls;
    Ntrain[i] = (int) (np.size(Data_trn0)/np.size(Data_trn0[0]));
    print "-- %d : rmsEr is %d for N %d --" %(i,mean[i],Ntrain[i]);   
  
    Outp[i] = Out_predtst;
    Outv[i]= np.transpose(Out_tst);
    if(i<(Nperc-1)):
        Data_trn0, Data_rem0 = add_randomTrn(Data_rem0,Data_trn0,sizeTrn);

plt.subplot(3,1,1)    
plt.plot((Outv[0]-Outp[0])/Outv[0],'o-r')   
plt.title("Training set of %d data points"%(Ntrain[0]))
plt.ylim((-8,8))
plt.xlim((0,100))
plt.subplot(3,1,2)
plt.plot((Outv[1]-Outp[1])/Outv[1],'o-r')   
plt.title("Training set of %d data points"%(Ntrain[2]))
plt.ylim((-8,8))
plt.xlim((0,100))
plt.ylabel('$(y-y_{pred})/y}$')
#plt.subplot(4,1,3)
#plt.plot((Outv[2]-Outp[2])/Outv[2],'o-r')  
#plt.title("Training set of 1800 data points")%(Ntrain[4]))
#plt.ylim((-8,8))
#plt.xlim((0,100))
plt.subplot(3,1,3)
plt.plot((Outv[6]-Outp[6])/Outv[6],'o-r')  
plt.title("Training set of %d data points"%(Ntrain[4]))
plt.ylim((-8,8))
plt.xlim((0,100))
plt.xlabel('instance')
#plt.plot(Outv[0],'o-r')    
#plt.plot(Outp[0],'o-b')
j=0
N=4
#for i in range(Nperc/N):
#    plt.subplot(Nperc/N+1,1,i+2)
#    plt.plot(Outv[j],'o-r')    
#    plt.plot(Outp[j],'o-b')
#    j=j+N
plt.show()

plot_hist_err(Outv,Outp,[0,1,6],100.)



if meth==0:
    diff,Out_trnya,Out_predtrnya, rmsya  = evaluate_tree(Data_trn0,indecisInp,indexOut)
if meth==1:
    diff,Out_trnya,Out_predtrnya, rmsya  = evaluate_poly(Data_trn0,indecisInp,indexOut,npoly)

plt.plot(diff,'o-r')
plt.xlabel('instance')
plt.ylabel('$(y-y_{pred})/y$')
plt.show()
binsh = np.arange(1200)/1200.-0.5;
hist1,hist2 = np.histogram(diff,bins = binsh)

binsh = np.arange(1199)/1200.-0.5;
plt.plot(np.transpose(binsh),hist1)
plt.show()



for i in range(Nperc):
    stnd = 0;
    for j in range(NprTrls):
        stnd = stnd+(error[i][j] - mean[i])*(error[i][j] - mean[i]);
    std[i] = np.sqrt(stnd/NprTrls);


plt.figure()
plt.errorbar(Ntrain,mean,yerr=std,fmt='o')
#plt.ploterr(Out_predtrn,'+r')

plt.xlabel('Number of training data points')
plt.ylabel('RMS percentage error ')
plt.show()

for i in range(Nperc):
    plt.plot(Ntrain[i]*np.ones(np.size(error[0])),error[i],'o')
#plt.plot(Out_predtst,'+b')

plt.xlabel('Number of training data points')
plt.ylabel('RMS percentage error ')
plt.show()



