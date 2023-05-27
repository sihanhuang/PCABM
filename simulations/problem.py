import argparse 
import numpy as np 
import sys
sys.path.insert (0, '..') 
import pcabm.sc as sc 
import pcabm.commFunc as cf 
import pcabm.plem as plem 
import pcabm.dcbm as dc 
import pcabm.pcabm as pcabm 
import pcabm.ecv as ecv 
import pandas as pd 
from scipy.optimize import minimize 
from sklearn.metrics.cluster import adjusted_rand_score 
from sklearn.metrics import accuracy_score


def gen_adj_5cov(n,k,p,rho,gamma):

    ####################################
    ## Generate Covariates
    ####################################
    Z = np.zeros((p,n,n))
    Z[0,:,:] = np.random.binomial(1, 0.1, size=(n,n))
    Z[1,:,:] = np.random.poisson(0.1, size=(n,n))
    Z[2,:,:] = np.random.uniform(0, 1, size=(n,n))
    Z[3,:,:] = np.random.exponential(0.3, size=(n,n))
    Z[4,:,:] = np.random.normal(0, 0.3, size=(n,n))
    Z = np.tril(Z, -1)
    Z = Z + np.transpose(Z, axes= (0,2,1))
    Z = np.transpose(Z,axes= (1,2,0))

    ####################################
    ## Generate Edges
    ####################################
    k = int(k)
    gt = np.random.randint (k, size=n) # ground truth
    B = np.ones ((n, n))
    for i in range(k):
        loc = np.where(gt==i)[0]
        B[np.ix_(loc, loc)]=2
    B=B*rho
    lam = np.multiply(B,np.exp(np.dot (Z, gamma) ))
    A = np. random.poisson(np.tril(lam, -1)); Ab = A+A.T

    return Ab,Z,gt

def gen_adj_2fake(n, k, rho, gamma, r):
    
    ####################################
    ## Generate Covariates
    ####################################
    Z = np.zeros((1, n, n))
    Z[0, : ,:] = np.random.poisson(0.09, size=(n, n))
    Z = np.tril(Z, -1)
    Z = Z + np.transpose(Z, axes=(0,2,1))
    Z = np.transpose (Z, axes= (1,2,0))

    ####################################
    ## Generate Fake Covariates
    ####################################
    B = np.ones ((n,n)) ;B[0: (n//2) ,0: (n//2)]=2;B[ (n//2) :n, (n//2) :n]=2;
    Z_fake = np.zeros( (1, n, n))
    Z_fake[0, :,:] = np.random.poisson (0.09, size=(n,n)) + 0.6*r*(1-r**2)**(-0.5)*(B-1.5)
    Z_fake = np.tril(Z_fake, -1)
    Z_fake = Z_fake+np.transpose(Z_fake, axes=(0,2,1))
    Z_fake = np.transpose(Z_fake, axes= (1, 2, 0))
    Z_both = np.concatenate((Z,Z_fake),axis=2)

    ####################################
    ## Generate Edges
    ####################################
    B=B*rho;
    lam = np.multiply (B, np.exp(Z*gamma)[ : , : ,0])
    A = np.random.poisson(np.tril(lam, -1)) ; Ab = A+A.T
    gt=np.array ([0]* (n//2)+[1]* (n//2)) # ground truth

    return Ab,Z,Z_fake,Z_both,gt

def solve (param):
    n = int(param['n']); #number
    rho = param['rho' ]*np.log(n)/n#sparsity
    gamma = param['gamma']*np.array ((0.4,0.8,1.2,1.6,2)) #signals of individual information
    k, p = 2, 5
    np.random.seed(int(param['seed']))

    Ab, Z, gt = gen_adj_5cov(n,k,p,rho,gamma)

    ####################################
    ## Estimate gamma
    ####################################

    gamma_est = minimize(cf.nLLGamma,np.zeros(p),args=(np.random.randint(2,size=n),Ab,Z,),
        method='BFGS', options={'disp': False}, tol=10e-5).x


    ####################################
    ## Estimate community
    ####################################

    ## SC without adjustment
    _,_, model_SC = sc.SC_r(Ab, 2)

    ## SC with adjustment
    _,_, model_SCWA = sc.SC_r(Ab/np.exp(np.dot(Z, gamma_est)),2)

    ## SBM MLE
    model_SBM = pcabm.PCABM(Ab, np.zeros(shape = (n,n, p)), 2, np.ones(p))
    sc_label = np.copy(model_SC.labels_)
    estSBM,_ = model_SBM.fit(community_init=sc_label, gt=gt, init = 50, tabu_size=n//3)

    ## PCABM MLE with SCWA as initial
    model_PCA = plem.PLEM(Ab, Z, 2, gamma_est)
    scwa_label = np.copy(model_SCWA.labels_)
    estPCA,_ = model_PCA.fit(community_init=scwa_label, gt=gt)

    ## DCBM
    model_DCBM = dc.DCBM(Ab, 2)
    sc_label = np.copy(model_SC.labels_)
    estDCBM, _ = model_DCBM.fit(community_init=sc_label,gt=gt, init = 50, tabu_size=n//3)

    return (adjusted_rand_score (model_SC.labels_, gt), adjusted_rand_score (model_SCWA.labels_,gt),
        adjusted_rand_score(estSBM, gt), adjusted_rand_score(estPCA, gt), adjusted_rand_score(estDCBM, gt))

def solve_gamma (param):
    n = int(param['n']); #number
    rho = param['rho' ]*np.log(n)/n#sparsity
    gamma = param['gamma']*np.array ((0.4,0.8,1.2,1.6,2)) #signals of individual information
    k, p = 2, 5
    np.random.seed(int(param['seed']))

    Ab, Z, gt = gen_adj_5cov(n,k,p,rho,gamma)

    ####################################
    ## Estimate gamma
    ####################################

    gamma_est = minimize(cf.nLLGamma,np.zeros(p),args=(np.random.randint(2,size=n),Ab,Z,),
        method='BFGS', options={'disp': False}, tol=10e-5).x

    return gamma_est

def solve_K(param):
    n = int(param['n']); #number
    rho = param['rho' ]*np.log(n)/n#sparsity
    gamma = param['gamma']*np.array ((0.4,0.8,1.2,1.6,2)) #signals of individual information
    k, p = param['k'], 5
    np.random.seed(int(param['seed']))

    Ab, Z, gt = gen_adj_5cov(n,k,p,rho,gamma)

    ####################################
    ## Estimate K
    ####################################

    res = ecv.chooseK(Ab,Z,k,)
    ans = res.fit()

    return ans
    
def shuffle(gt, acc) :
    err = 1-acc;
    num = int(len(gt)*err/2)
    shuf1 = np.random.choice(len(gt)//2,size=num,replace=False)
    shuf2 = len(gt)//2+np.random.choice(len(gt)//2,size=num,replace=False)
    gt_ret = gt.copy()
    gt_ret[shuf1]=1-gt_ret[shuf1];gt_ret[shuf2]=1-gt_ret[shuf2];
    return gt_ret

def solve_init (param):
    n = int(param['n']); #number
    rho = param['rho' ]*np.log(n)/n#sparsity
    gamma = param['gamma']*np.array ((0.4,0.8,1.2,1.6,2)) #signals of individual information
    k, p = 2, 5
    np.random.seed(int(param['seed']))

    Ab, Z, gt = gen_adj_5cov(n,k,p,rho,gamma)

    ####################################
    ## Estimate gamma
    ####################################

    gamma_est = minimize(cf.nLLGamma,np.zeros(p),args=(np.random.randint(2,size=n),Ab,Z,),
        method='BFGS', options={'disp': False}, tol=10e-5).x


    ####################################
    ## Estimate community
    ####################################

    ## SC with adjustment
    _,_, model_SCWA = sc.SC_r(Ab/np.exp(np.dot (Z, gamma_est)), 2)

    ## PCABM MLE with shuffled initial
    model_PCA = plem.PLEM(Ab, Z, 2, gamma_est)
    scwa_label = np.copy(model_SCWA.labels_)
    estPCA,_ = model_PCA.fit(community_init=shuffle(gt,param['shuffle']), gt=gt)

    return (max(accuracy_score(model_SCWA.labels_,gt), accuracy_score(model_SCWA.labels_,1-gt)),
        max(accuracy_score(estPCA,gt),accuracy_score(estPCA,1-gt)))

def solve_dcbm(param) :
    n = int(param['n']) #number
    rho = param['rho' ]*np.log(n)/n #sparsity 
    k, p = 2, 1
    np.random.seed(int(param['seed']))

    ####################################
    ## Estimate covariates
    ####################################
    theta = 1+3*np.random.binomial (1,0.5, size=n)

    ####################################
    ## Estimate edges
    ####################################
    B = np.ones((n,n));B[0:(n//2) , 0:(n//2)]=2;B[(n//2):n, (n//2):n]=2;B=B*rho
    A = np.random.poisson(np.tril(B*np.outer(theta,theta), -1)); Ab = A+A.T
    gt = np.array([0]* (n//2)+ [1]* (n//2)) # ground truth

    dvec = np.sum(Ab ,axis = 0)
    dvec_mat = np.array( [dvec, ]*n)
    Z = np.log(dvec_mat)[ ..., np.newaxis]+np.log(dvec_mat.transpose()) [ ..., np.newaxis]

    ####################################
    ## Estimate gamma
    ####################################

    gamma_est = minimize(cf.nLLGamma,np.zeros(p),args=(np.random.randint(2,size=n),Ab,Z,),
        method='BFGS', options={'disp': False}, tol=10e-5).x


    ####################################
    ## Estimate community
    ####################################

    ## SC without adjustment
    _,_, model_SC = sc.SC_r(Ab, 2)

    ## SC with adjustment
    _,_, model_SCWA = sc.SC_r(Ab/np.exp(np.dot(Z, gamma_est)),2)

    ## SBM MLE
    model_SBM = pcabm.PCABM(Ab, np.zeros(shape = (n,n, p)), 2, np.ones(p))
    sc_label = np.copy(model_SC.labels_)
    estSBM,_ = model_SBM.fit(community_init=sc_label, gt=gt, init = 50, tabu_size=n//3)

    ## PCABM MLE with SCWA as initial
    model_PCA = plem.PLEM(Ab, Z, 2, gamma_est)
    scwa_label = np.copy(model_SCWA.labels_)
    estPCA,_ = model_PCA.fit(community_init=scwa_label, gt=gt)

    ## DCBM
    model_DCBM = dc.DCBM(Ab, 2)
    sc_label = np.copy(model_SC.labels_)
    estDCBM, _ = model_DCBM.fit(community_init=sc_label,gt=gt, init = 50, tabu_size=n//3)

    ## SCORE
    model_SCORE = sc.SCORE(Ab,2)

    return (adjusted_rand_score (model_SC.labels_, gt), adjusted_rand_score (model_SCWA.labels_,gt),
        adjusted_rand_score(estSBM, gt), adjusted_rand_score(estPCA, gt), adjusted_rand_score(estDCBM, gt),
        adjusted_rand_score(model_SCORE.labels_, gt))







def solve_vs(param):
    n = int(param['n']); #number
    rho = param['rho' ]*np.log(n)/n#sparsity
    gamma = param['gamma'] #signals of individual information
    r = param['corr']
    k, p = 2, 2
    np.random.seed(int(param['seed']))

    Ab, Z, Z_fake, Z_both, gt = gen_adj_2fake(n,k,rho,gamma,r)

    ####################################
    ## Select variable
    ####################################
    vs = ecv.variableSelect(Ab, Z_both, k)
    selected_cov = vs.fit()


    ####################################
    ## Estimate gamma
    ####################################

    gamma_t = minimize(cf.nLLGamma, np.zeros (1) ,args=(np.random.randint (2, size=n) ,Ab, np.expand_dims(Z_both[:,:,0], axis=2),),
        method='BFGS', options={'disp': False}, tol=10e-5).x
    gamma_f = minimize(cf.nLLGamma, np.zeros (1) ,args=(np.random.randint (2, size=n) ,Ab, np.expand_dims(Z_both[:,:,1], axis=2),),
        method='BFGS', options={'disp': False}, tol=10e-5).x
    gamma_b = minimize(cf.nLLGamma, np.zeros (p), args=(np.random.randint (2, size=n) ,Ab, Z_both,),
        method='BFGS', options={'disp': False}, tol=10e-5).x
    gamma_s = minimize(cf.nLLGamma, np.zeros (len(selected_cov)),args=(np.random.randint (2, size=n),Ab,Z_both[:, :,selected_cov],),
        method='BFGS', options={'disp': False}, tol=10e-5).x


    ####################################
    ## Estimate community
    ####################################

    _,_,model_SCWA_n = sc.SC_r(Ab, 2)

    ## SC with adjustment
    _,_,model_SCWA_t = sc.SC_r(Ab/np.exp(np.dot(np.expand_dims (Z_both[:, : ,0], axis=2), gamma_t)),2)
    _,_,model_SCWA_f = sc.SC_r(Ab/np.exp(np.dot(np.expand_dims (Z_both[:, : ,1], axis=2), gamma_f)),2)
    _,_,model_SCWA_b = sc.SC_r(Ab/np.exp(np.dot(Z_both, gamma_b)), 2)
    if len(selected_cov) == 1:
        _,_,model_SCWA_s = sc.SC_r(Ab/np.exp(np.dot(np.expand_dims (Z_both[ : , : ,selected_cov[0]], axis=2), gamma_s)), 2)
    else:
        _,_,model_SCWA_s = sc.SC_r(Ab/np.exp(np.dot(Z_both,gamma_s)), 2)


    return (adjusted_rand_score(model_SCWA_t.labels_,gt),adjusted_rand_score(model_SCWA_f.labels_,gt),
        adjusted_rand_score(model_SCWA_b.labels_,gt),adjusted_rand_score(model_SCWA_s.labels_,gt),
        adjusted_rand_score(model_SCWA_n.labels_,gt))

def solve_cov(param):
    n = int(param['n']); #number
    rho = param['rho' ]*np.log(n)/n#sparsity
    gamma = param['gamma'] #signals of individual information
    r = param['corr']
    k, p = 2, 2
    np.random.seed(int(param['seed']))

    Ab, Z, Z_fake, Z_both, gt = gen_adj_2fake(n,k,rho,gamma,r)

    ####################################
    ## Select variable
    ####################################
    vs = ecv.variableSelect(Ab, Z_both, k)
    selected_cov = vs.fit()

    return selected_cov

