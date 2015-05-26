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




##############################

Nperc = 15;
NprTrls = 15;

error = np.zeros((Nperc,NprTrls));
mean = np.zeros(Nperc);
std = np.zeros(Nperc);
Ntrain = np.zeros(Nperc);


percentageTrn0 = 90/Nperc;    
percentageTst0 = 5;
Data_trn0, Data_tst0, Data_rem0 = random_TrnTst(Data_shuf,percentageTrn0,percentageTst0);
sizeTrn = (int) (np.size(Data_M)*percentageTrn0/100)-1;
sizeTst = (int) (np.size(Data_M)*percentageTst0/100)-1;


for i in range(Nperc):
    mn = 0;
    Data_trn0, Data_rem0 = add_randomTrn(Data_rem0,Data_trn0,sizeTrn);
    for j in range(NprTrls):
        Data_tst0 = get_randTest(Data_rem0,sizeTst)
        Out_trn,Out_predtrn, Out_tst,  Out_predtst, rmsEr =find_pred_poly(Data_trn0,Data_tst0,indecisInp,indexOut,npoly);
        error[i][j] = rmsEr;
        mn = mn + rmsEr;
    
    mean[i] = mn/NprTrls;
    Ntrain[i] = np.size(Data_trn0)/np.size(Data_trn0[0]);
    print mean[i]

for i in range(Nperc):
    stnd = 0;
    for j in range(NprTrls):
        stnd = stnd+(error[i][j] - mean[i])*(error[i][j] - mean[i]);
    std[i] = np.sqrt(stnd/NprTrls);


plt.figure()
plt.errorbar(Ntrain,mean,yerr=std,fmt='o')
#plt.ploterr(Out_predtrn,'+r')

plt.show()

for i in range(Nperc):
    plt.plot(Ntrain[i]*np.ones(np.size(error[0])),error[i],'o')
#plt.plot(Out_predtst,'+b')

plt.show()



