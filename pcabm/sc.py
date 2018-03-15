import numpy as np
import pandas as pd
from numpy import linalg as LA
import pcabm.commFunc as cf
from sklearn.cluster import KMeans
from scipy.optimize import minimize


def SC(Ab,k):
    L = pd.DataFrame(Ab)
    w, v = LA.eig(L)
    w.argsort()[-k:][::-1]
    X = v[:,w.argsort()[-k:][::-1]]
    kmeans = KMeans(n_clusters=k).fit(np.real(X))
    return(kmeans)

def SCORE(Ab,k):
    n=Ab.shape[0]
    L = pd.DataFrame(Ab)
    w, v = LA.eig(L)
    X = v[:,abs(w).argsort()[-k:][::-1]]
    Y = X[:,1]/X[:,0];Y=np.nan_to_num(Y);Y=np.real(Y)#;Y[abs(Y)>100]=0
    kmeans = KMeans(n_clusters=k).fit(Y.reshape((n,k-1)))
    return(kmeans)


def SCWA(Ab,Z,k,coef=1):
    p = Z.shape[2]
    n = Ab.shape[0]
    gamma = minimize(cf.nLLGamma,np.zeros(p),args=(np.repeat([0,1], n//2),Ab,Z,),method='BFGS', options={'disp': False}, tol=10e-5).x
    
    indi = coef*np.ones(n*n); indi = indi.reshape((n,n));adjust = np.maximum(indi,np.exp(np.dot(Z,gamma)))
    L = pd.DataFrame(Ab/adjust)
    w, v = LA.eig(L)
    X = v[:,abs(w).argsort()[-k:][::-1]]
    kmeans = KMeans(n_clusters=k).fit(X)
    return(kmeans,gamma)

def SCWAd(Ab,Z,k,coef=1):
    n = Ab.shape[0]
    p = Z.shape[2]
    gamma = minimize(cf.nLLGamma,np.zeros(p),args=(np.repeat([0,1], n//2),Ab,Z,),method='BFGS', options={'disp': False}, tol=10e-5).x
    
    indi = coef*np.ones(n*n); indi = indi.reshape((n,n));adjust = np.maximum(indi,np.exp(np.dot(Z,gamma)))
    L = pd.DataFrame(Ab/adjust)
    w, v = LA.eig(L)
    X = v[:,abs(w).argsort()[-k:][::-1]]
    Y = X[:,1]/X[:,0]
    kmeans = KMeans(n_clusters=k).fit(Y.reshape((n,k-1)))
    return(kmeans)

