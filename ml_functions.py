import csv
import math
import operator
import scipy as sp
import sys
import time


import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn import tree
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import svm
#from ml_funcs import *



def GetData(Gfname):
    with open(Gfname, 'rU') as f:
        reader = csv.reader(f)
        Data = [['0']];
        f.readline();
        for row in reader:
            Data.append(row)
    
    Data = np.delete(Data,(0),axis=0)

    sizeD = np.size(Data);  
    for i in range(sizeD):
        for j in range(np.size(Data[1])):
            Data[i][j] = (float) (Data[i][j]);
    i=0


    return Data



def shuffle(Data,Nth): # Nth best as 512
    decks = np.size(Data)/Nth;  
    Data_shuff = [['0']];   
    for j in range(Nth):
        for i in range(decks):
            Data_shuff.append(Data[j+i*512])
    
    Data_shuff = np.delete(Data_shuff,(0),axis=0)
    return Data_shuff
 

def random_TrnTst_comp(Data,percentageTrn):
    
    
    if percentageTrn>100:
        sys.exit('the sum of the percetnages cant be more than 100%!');
    
    sizeTrn = (int) (np.size(Data)*percentageTrn/100.)-1;
    
    
    idx = [(int) (random.random()*np.size(Data))];
    rndDtTrng = [Data[idx[np.size(idx)-1]]]
    Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    for i in range(sizeTrn):
        idx.append((int) (random.random()*np.size(Data)));
        rndDtTrng.append(Data[idx[np.size(idx)-1]])
        Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    
    

    return rndDtTrng, Data


def get_randtrn(Data,sizeTrn):
    
    
    if sizeTrn>np.size(Data):
        sys.exit('too big a size!');
    
    
    idx = [(int) (random.random()*np.size(Data))];
    rndDtTrng = [Data[idx[np.size(idx)-1]]]
    Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    for i in range(sizeTrn):
        idx.append((int) (random.random()*np.size(Data)));
        rndDtTrng.append(Data[idx[np.size(idx)-1]])
        Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    
    
    
    return rndDtTrng, Data




def add_randomTrn(Data,DataTrn,sizeTrn):
    
    if sizeTrn>np.size(Data):
        sys.exit('number exceeds data points');
 
  
    idx = [(int) (random.random()*np.size(Data)/np.size(Data[0]))];

            #    DataTrn.append(Data[idx[0]])
            #   Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    for i in range(sizeTrn-1):
        idx.append((int) (random.random()*np.size(Data)/np.size(Data[0])));
        DataTrn.append(Data[idx[np.size(idx)-1]])
        Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)

    return DataTrn, Data 


def get_randTest(Data,sizeTst):
    if sizeTst>np.size(Data):
       sys.exit('number exceeds data points');
    
    
    idx = [(int) (random.random()*np.size(Data)/np.size(Data[0]))];
    DataTst = [Data[idx[np.size(idx)-1]]]
    Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    for i in range(sizeTst-1):
        idx.append((int) (random.random()*np.size(Data)/np.size(Data[0])));
        DataTst.append(Data[idx[np.size(idx)-1]])
        Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    
    return DataTst


def GetInput(Data,indecis):
    NumInp = np.size(indecis);
    if NumInp > (np.size(Data[0])-1):
        print NumInp
        print Data[0]
        sys.exit('you assumed too many inputs! impossiaable!');
    
    for i in indecis:
        if i>(np.size(Data[0])):
            print i
            sys.exit('one of your assumed indecis for input is out of bound!! impossiaable!');
    
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
            sys.exit('one of your assumed indecis of output is out of bound!! impossiaable!');
    
    Nx = np.size(Data)/np.size(Data[0]); 
    Ny = (int) (NumOut);
    Out = np.zeros((Nx,Ny))
    
    for i in range(Nx):
        k=0;
        for j in indecis:
            Out[i][k] = Data[i][j];
            k=k+1;
    
    return Out




def find_pred_tree(DataTrn,DataTst,indecisInp,indexOut):
    
    Inp_trn = GetInput(DataTrn,indecisInp);
    Out_trn = GetOutput(DataTrn,indexOut);

    
    Inp_tst = GetInput(DataTst, indecisInp);
    Out_tst = GetOutput(DataTst, indexOut);
    
    clf = tree.DecisionTreeRegressor()
    clf.fit(Inp_trn, Out_trn) 
    
    
    Out_predtrn = np.ones(np.size(Out_trn));
    indx = 0;
    for tstinp in Inp_trn:
        Out_predtrn[indx] = clf.predict(tstinp);
        indx = indx+1;
    
    
    Out_predtst = np.ones(np.size(Out_tst));
    indx=0;
    for tstinp in Inp_tst:
        Out_predtst[indx] = clf.predict(tstinp);
        indx = indx+1;
    
    ms = 0;
    diff = np.ones(np.size(Out_tst));
    for i in range(np.size(Out_tst)):
        diff[i] = (Out_tst[i]-Out_predtst[i])*(Out_tst[i]-Out_predtst[i])#/(Out_tst[i]*Out_tst[i])
        ms = ms+diff[i];
    
    
    rms = 100*np.sqrt(ms/np.size(Out_tst))/(max(Out_tst)-min(Out_tst));
    
    return Out_trn,Out_predtrn, Out_tst,  Out_predtst, rms 




def find_pred_poly(DataTrn,DataTst,indecisInp,indexOut,npoly):


    Inp_trn = GetInput(DataTrn,indecisInp);
    Out_trn = GetOutput(DataTrn,indexOut);
    

    Inp_tst = GetInput(DataTst, indecisInp);
    Out_tst = GetOutput(DataTst, indexOut);

    #clf = make_pipeline(PolynomialFeatures(npoly), LinearRegression())
    clf = make_pipeline(PolynomialFeatures(npoly), Ridge(alpha = 0.1))
    clf.fit(Inp_trn, Out_trn) 
    
    
    Out_predtrn = np.ones(np.size(Out_trn));
    indx = 0;
    for tstinp in Inp_trn:
        Out_predtrn[indx] = clf.predict(tstinp);
        indx = indx+1;
    


    Out_predtst = np.ones(np.size(Out_tst));
    indx=0;
    for tstinp in Inp_tst:
        Out_predtst[indx] = clf.predict(tstinp);
        indx = indx+1;
    ms = 0;
    diff = np.ones(np.size(Out_tst));
    for i in range(np.size(Out_tst)):
        diff[i] = (Out_tst[i]-Out_predtst[i])*(Out_tst[i]-Out_predtst[i])#/(Out_tst[i]*Out_tst[i])
        ms = ms+diff[i];
    

    rms = 100*np.sqrt(ms/np.size(Out_tst))/(max(Out_tst)-min(Out_tst));

    return Out_trn,Out_predtrn, Out_tst,  Out_predtst, rms 






def plot_hist_err(Outv,Outp,arr,Nbin,meth):
    N = np.size(arr)
        
    diff = (Outv[arr[2]]-Outp[arr[2]])/Outv[arr[2]]
    widb =(diff.max()-diff.min())/Nbin
    for i in range(N):
        diff = (Outv[arr[i]]-Outp[arr[i]])/Outv[arr[i]]
        #bin = np.arange(diff.min(),diff.max(),(diff.max()-diff.min())/Nbin)
        bin = np.arange(diff.min(),diff.max(),widb)
        plt.subplot(N,1,i+1)  
        hist,b = np.histogram(diff,bins = bin )
        #bin = bin.tolist()
        width = 0.7 * (b[1] - b[0])
        center = (b[:-1] + b[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
    #        plt.plot(center,hist)
        plt.xlim((-3,3))
        if i == N-1:
            plt.xlabel('$(y-y_{pred})/y$')
        if (i == 0 and meth==0): 
            plt.title('histogram for decision tree regression')
        if (i == 0 and meth==1): 
                plt.title('histogram for polynomial regression')

        
    plt.show()


def plot_hist_err2(Outv,Outp,arr,Nbin):
    N = np.size(arr)
    for i in range(N):
        diff = (Outv[arr[i]]-Outp[arr[i]])/Outv[arr[i]]
        plt.subplot(N,1,i+1)  
        Ed = 2.
        bin = np.arange(-1*Ed,Ed,(2.*Ed)/((float)(Nbin)))
        hist,b = np.histogram(diff,bins = bin)
        #bin = bin.tolist()
        width = 0.7 * (b[1] - b[0])
        center = (b[:-1] + b[1:]) / 2
        plt.bar(center, hist, align='center', width=width)        
        plt.xlim((-2,2))

        if i == N-1:
                plt.xlabel('$(y-y_{pred})/y$')
        if i == 0: 
            plt.title('histogram for forth order polynomial regression')
    

   
    plt.show()




def plot_hist_err_hist(Outv,Outp,arr,Nbin):
    N = np.size(arr)
    for i in range(N):
        diff = (Outv[arr[i]]-Outp[arr[i]])/Outv[arr[i]]
        plt.subplot(N,1,i+1)  
        hist,bins = np.histogram(diff,bins = Nbin,density = True)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist/((float)(np.size(Outv[0]))), align='center', width=width)

    plt.show()






