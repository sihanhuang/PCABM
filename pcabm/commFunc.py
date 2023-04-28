 # Community Related functions
 import numpy as np

 def position(e):
     pos=[]
     for i in np.arange(e.max()+1):
         pos.append(np.flatnonzero(e==i))
     return pos

 def summatrix(O):
     Ov = sum(O)
     return Ov[:,None]*Ov

 def num(e):
     _, num = np.unique(e, return_counts=True)
     return num


 def O(e,Ab):
     O = np.zeros((e.max()+1,e.max()+1))
     for i in np.arange(e.max()+1):
         for j in np.arange(e.max()+1):
             posi=position(e)
             O[i,j]= np.sum(Ab[np.ix_(posi[i],posi[j])])
     return O

 def E(e,gamma,Z):
     if type(gamma)==float or type(gamma)==int or type(gamma)==np.float64 or type(gamma)==np.int64:
         gamma = np.array([gamma])

     K = e.max()
     exp_mat = np.exp(np.dot(Z,gamma))
     posi=position(e)
     E = np.zeros((K+1,K+1))

     for i in np.arange(K+1):
         for j in np.arange(i+1,K+1):
             E[i,j] = np.sum(exp_mat[np.ix_(posi[i],posi[j])])
     E = E+E.T

     for i in np.arange(K+1):
         E[i,i]= np.sum(exp_mat[np.ix_(posi[i],posi[i])])-np.sum(exp_mat[posi[i],posi[i]])
     return E


 def E1(e,gamma,Z):   ## return K*K*p
     if type(gamma)==float or type(gamma)==int or type(gamma)==np.float64 or type(gamma)==np.int64:
         gamma = np.array([gamma])

     K = e.max()

     exp_mat = np.exp(np.dot(Z,gamma))  # K*K

     posi=position(e)
     E = np.zeros((K+1,K+1,Z.shape[2]))

     for i in np.arange(K+1):
         for j in np.arange(i+1,K+1):

             E[i,j,:] = np.tensordot(exp_mat[np.ix_(posi[i],posi[j])],Z[np.ix_(posi[i],posi[j])])
     E = E+np.transpose(E,axes=(1,0,2))

     for i in np.arange(K+1):
         E[i,i,:]= np.tensordot(exp_mat[np.ix_(posi[i],posi[i])],Z[np.ix_(posi[i],posi[i])])-np.dot(exp_mat[posi[i],posi[i]],Z[posi[i],posi[i],:])
     return E



 def nLL(e,gamma,Ab,Z):
     if type(gamma)==float or type(gamma)==int or type(gamma)==np.float64 or type(gamma)==np.int64:
         gamma = np.array([gamma])
     A = np.tril(Ab,-1)
     n = A.shape[0]
     return (np.sum(O(e,Ab)*np.log(E(e,gamma,Z)))/2-np.nansum(O(e,Ab)*np.log(O(e,Ab))/2-num(e)*np.log(num(e)/n)))/(n**2)

 def nLLGamma(gamma,e,Ab,Z):
     if type(gamma)==float or type(gamma)==int or type(gamma)==np.float64 or type(gamma)==np.int64:
         gamma = np.array([gamma])
     n = Ab.shape[0]
     A = np.tril(Ab,-1)
     return (np.sum(O(e,Ab)*np.log(E(e,gamma,Z)))/2-np.sum(A*np.dot(Z,gamma)))/(n**2)

 def gamma_der(gamma,e,Ab,Z):
     if type(gamma)==float or type(gamma)==int or type(gamma)==np.float64 or type(gamma)==np.int64:
         gamma = np.array([gamma])
     n = Ab.shape[0]
     A = np.tril(Ab,-1)
     return (np.tensordot(O(e,Ab),E1(e,gamma,Z)/E(e,gamma,Z)[:,:,None])/2-np.tensordot(A,Z))/(n**2)


 def Infomatrix(Z,gamma):
     exp_mat = np.exp(np.dot(Z,gamma))
     n = Z.shape[0];N=n*(n-1)
     v0 = np.sum(np.exp(np.dot(Z,gamma)))/N
     v1 = np.tensordot(exp_mat,Z,axes=([0,1],[0,1]))/N
     v2 = np.tensordot(exp_mat,Z[...,None]*Z[...,None,:],axes=([0,1],[0,1]))/N
     return v2-np.outer(v1,v1)/v0

 def Info(e,gamma,Ab,Z):
     n=Z.shape[0];N=n*(n-1)/2
     pi=np.array([len(i) for i in position(e)])/n
     return np.sum(np.outer(pi,pi)*O(e,Ab)/E(e,gamma,Z))*N*Infomatrix(Z,gamma)