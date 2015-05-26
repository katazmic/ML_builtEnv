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
#from ml_funcs import *



def GetData():
    Gfname = 'embodied_SI.csv';
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
    return Data



def shuffle(Data,Nth): # Nth best as 512
    decks = np.size(Data)/Nth;  
    Data_shuff = [['0']];   
    for j in range(Nth):
        for i in range(decks):
            Data_shuff.append(Data[j+i*512])
    
    Data_shuff = np.delete(Data_shuff,(0),axis=0)

    return Data_shuff
 


def random_TrnTst(Data,percentageTrn,percentageTst):
    
    
    if percentageTrn+percentageTst>100:
        sys.exit('the sum of the percetnages cant be more than 100%!');
    
    sizeTrn = (int) (np.size(Data)*percentageTrn/100)-1;
    sizeTst = (int) (np.size(Data)*percentageTst/100)-1;
   
    
    idx = [(int) (random.random()*np.size(Data))];
    rndDtTrng = [Data[idx[np.size(idx)-1]]]
    Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    for i in range(sizeTrn):
        idx.append((int) (random.random()*np.size(Data)));
        rndDtTrng.append(Data[idx[np.size(idx)-1]])
        Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    
    idx = [(int) (random.random()*np.size(Data))];
    rndDtTst = [Data[idx[np.size(idx)-1]]]
    Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    for i in range(sizeTst):
        idx.append((int) (random.random()*np.size(Data)));
        rndDtTst.append(Data[idx[np.size(idx)-1]])
        Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    
    
    return rndDtTrng, rndDtTst, Data


def add_randomTrn(Data,DataTrn,sizeTrn):
    if sizeTrn>np.size(Data):
        sys.exit('number exceeds data points');
 
    
    idx = [(int) (random.random()*np.size(Data))];
    DataTrn.append(Data[idx[np.size(idx)-1]])
    Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    for i in range(sizeTrn):
        idx.append((int) (random.random()*np.size(Data)));
        DataTrn.append(Data[idx[np.size(idx)-1]])
        Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)

    return DataTrn, Data 


def get_randTest(Data,sizeTst):
    if sizeTst>np.size(Data):
        sys.exit('number exceeds data points');
    
    
    idx = [(int) (random.random()*np.size(Data))];
    DataTst = [Data[idx[np.size(idx)-1]]]
    Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    for i in range(sizeTst):
        idx.append((int) (random.random()*np.size(Data)));
        DataTst.append(Data[idx[np.size(idx)-1]])
        Data = np.delete(Data,(idx[np.size(idx)-1]),axis=0)
    
    return DataTst




def GetInput(Data,indecis):
    NumInp = np.size(indecis);
    if NumInp > (np.size(Data[0])-1):
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




def find_pred_poly(DataTrn,DataTst,indecisInp,indexOut,npoly):


    Inp_trn = GetInput(DataTrn,indecisInp);
    Out_trn = GetOutput(DataTrn,indexOut);
    

    Inp_tst = GetInput(DataTst, indecisInp);
    Out_tst = GetOutput(DataTst, indexOut);

    clf = make_pipeline(PolynomialFeatures(npoly), Ridge())
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
        diff[i] = (Out_tst[i]-Out_predtst[i])*(Out_tst[i]-Out_predtst[i])/(Out_tst[i]*Out_tst[i])
        ms = ms+diff[i];
    

    rms = 100*np.sqrt(ms/np.size(Out_tst));

    return Out_trn,Out_predtrn, Out_tst,  Out_predtst, rms 




