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


from ml_functions import *




#### setting up the data to be analyzed

Data_M = GetData('embodied_SI_newSum.csv');
Data_shuf = Data_M; shuffle(Data_M,512);  # to shuffle around the data before taking a random sample

indecisInp = range(9);
indexOut = [9];
npoly = 4;

##############################

Nperc = 16;
NprTrls = 6;

error = np.zeros((Nperc,NprTrls));
mean = np.zeros(Nperc);
std = np.zeros(Nperc);
Ntrain = np.zeros(Nperc);

PercTrnIni = 70.

Data_trnI, Data_tstI= random_TrnTst_comp(Data_shuf,PercTrnIni);




sizeTst = (int) (np.size(Data_tstI)/10.);
sizeTrn = (int) (np.size(Data_trnI)/np.size(Data_trnI[0])/(Nperc+1));

print sizeTst
print sizeTrn

Outv = np.zeros((Nperc,sizeTst))
Outp = np.zeros((Nperc,sizeTst))

    


Data_trn = get_randTest(Data_trnI,sizeTrn)
Data_tst = get_randTest(Data_tstI,sizeTst)

Data_rem = Data_trnI;

for j in range(NprTrls):


    Data_trn = get_randTest(Data_trnI,sizeTrn)
    Data_tst = get_randTest(Data_tstI,sizeTst)
    Data_rem = Data_trnI;
    
    for i in range(Nperc):
        Out_trn,Out_predtrn, Out_tst,  Out_predtst, rmsEr =find_pred_poly(Data_trn,Data_tst,indecisInp,indexOut,npoly);
        error[i][j] = rmsEr;
    

        if j==0:
            Ntrain[i] =(int) (np.size(Data_trn)/np.size(Data_trn[0]))
        print "error for trial %d and N:%d  is %d-"%(j+1,(int) (np.size(Data_trn)/np.size(Data_trn[0])),error[i][j]);   
        if i<(Nperc-1):
            Data_trn, Data_rem = add_randomTrn(Data_rem,Data_trn,sizeTrn)




for i in range(Nperc):
    mean[i] = 0
    for j in range(NprTrls):
        mean[i] = error[i][j]+mean[i];
    mean[i] = mean[i]/NprTrls


for i in range(Nperc):
    stnd = 0;
    for j in range(NprTrls):
        stnd = stnd+(error[i][j] - mean[i])*(error[i][j] - mean[i]);
    std[i] = np.sqrt(stnd/NprTrls);

print 'see figure'
plt.figure()
plt.errorbar(Ntrain,mean,yerr=std,fmt='o')

plt.xlabel('Number of training data points')
plt.ylabel('Normalized RMS error (%)')

plt.title('Polynomial Regression')
plt.show()

for i in range(Nperc):
    plt.plot(Ntrain[i]*np.ones(np.size(error[0])),error[i],'o')


plt.xlabel('Number of training data points')
plt.ylabel('Normalized RMS error (%)')
plt.show()



