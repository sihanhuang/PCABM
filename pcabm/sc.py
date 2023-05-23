import numpy as np
import pandas as pd
from numpy import linalg as LA
import pcabm.commFunc as cf
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs


def SC(Ab,k):
    w, v = eigs(0.0+Ab, k, which='LR')
    kmeans = KMeans(n_clusters=k).fit(np.real(v))
    return(np.real(w),np.real(v),kmeans)

def SC_r(Ab,k):
    d = np.mean(Ab);
    dvec = np.mean(Ab, axis=0);
    adjvec = 2*d/dvec;
    adjvec[adjvec>1]=1;
    adjmat=np.sqrt(np.outer(adjvec,adjvec));
    
    return(SC(Ab*adjmat,k))

def SCLP(Ab,k):
    dvec = np.mean(Ab, axis=0);
    adjmat=np.sqrt(np.outer(1/dvec,1/dvec));

    w, v = eigs(0.0+Ab*adjmat, k, which='LR');
    kmeans = KMeans(n_clusters=k).fit(np.real(v));
    return(np.real(w),np.real(v),kmeans)

def SCLP_r(Ab,k,tau=0):
    if tau == 0:
        tau = np.max(np.mean(Ab, axis=0));
    Ab = tau*np.ones(Ab.shape[0])/Ab.shape[0]+Ab;
    return(SCLP(Ab,k))

def SCORE(Ab,k):
    n = Ab.shape[0]
    w, v = LA.eig(Ab)
    X = v[:,abs(w).argsort()[-k:][::-1]]
    Y = X[:,1]/X[:,0];Y=np.nan_to_num(Y);Y=np.real(Y)#;Y[abs(Y)>100]=0
    kmeans = KMeans(n_clusters=k).fit(Y.reshape((n,k-1)))
    return(kmeans)


def SCWA(Ab,Z,k, gamma=0, degree = False):
    if len(Z.shape)==2:
        p=1
    else:
        p = Z.shape[2]
    n = Ab.shape[0]
    if not np.array(gamma).all():
        gamma = minimize(cf.nLLGamma,np.zeros(p),args=(np.random.randint(2, size=n),Ab,Z,),method='BFGS', options={'disp': False}, tol=10e-5).x
    
    #indi = coef*np.ones(n*n); indi = indi.reshape((n,n));adjust = np.maximum(indi,np.exp(np.dot(Z,gamma)))
    adjust = np.exp(np.dot(Z,gamma))
    L = pd.DataFrame(Ab/adjust)
    w, v = LA.eig(L)
    X = np.real(v[:,abs(w).argsort()[-k:][::-1]])
    if degree:
        Y = X[:,1]/X[:,0]
        X = Y.reshape((n,k-1))
    kmeans = KMeans(n_clusters=k).fit(X)
    return(kmeans,gamma)


def SCWA_r(Ab,Z,k, gamma=0, degree = False):
    if len(Z.shape)==2:
        p=1
    else:
        p = Z.shape[2]
    n = Ab.shape[0]
    if not np.array(gamma).all():
        gamma = minimize(cf.nLLGamma,np.zeros(p),args=(np.random.randint(2, size=n),Ab,Z,),method='BFGS', options={'disp': False}, tol=10e-5).x
    
    adjust = np.exp(np.dot(Z,gamma))
    return SC_r(Ab/adjust,k)


