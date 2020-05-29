import numpy as np
import pandas as pd
import random
import pcabm.commFunc as cf
from scipy.optimize import minimize
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score



class PCABM():

    def __init__(self, Ab, Z, k, gamma=0):
        self.Ab = Ab
        self.Z = Z
        self.k = k
        self.n = Ab.shape[0]
        if len(self.Z.shape)==2:
            self.p = 1
        else:
            self.p = self.Z.shape[2]
        if np.array(gamma).all():
            self.gamma = gamma
        else:
            self.gamma = minimize(cf.nLLGamma,np.zeros(self.p),args=(np.random.randint(2, size=self.n),self.Ab,Z,),method='BFGS', options={'disp': False}, tol=10e-5).x
    

    def _updateO(self,oldO,oldcommunity,indice,newlabel):
        oldlabel = oldcommunity[indice]
        newO = np.copy(oldO)
        newcommunity = np.copy(oldcommunity)
        newcommunity[indice] = newlabel
        posi=cf.position(newcommunity)
        changeO = sum(self.Ab[indice,posi[oldlabel]])
        changeN = sum(self.Ab[indice,posi[newlabel]])
        for j in np.arange(newcommunity.max()+1):
            change = sum(self.Ab[indice,posi[j]])
            newO[newlabel,j] = newO[newlabel,j]+change+change*(newlabel==j)-changeN*(oldlabel==j)
            newO[oldlabel,j] = newO[oldlabel,j]-change-change*(oldlabel==j)+changeO*(newlabel==j)
        newO[:,newlabel] = np.transpose(newO[newlabel,:])
        newO[:,oldlabel] = np.transpose(newO[oldlabel,:])
        return newO

    def _updateE(self,oldE,oldcommunity,indice,newlabel):
        oldlabel = oldcommunity[indice]
        
        newE = np.copy(oldE)
        newcommunity = np.copy(oldcommunity)
        newcommunity[indice] = newlabel
        
        posi = cf.position(newcommunity)
        
        changeO = sum(np.exp(np.dot(self.Z[indice,posi[oldlabel],:],self.gamma)))   # just use self.gamma for now
        
        changeN = sum(np.exp(np.dot(self.Z[indice,np.setdiff1d(posi[newlabel],indice),:],self.gamma)))
        
        for j in np.arange(newcommunity.max()+1):
            if j==newlabel:
                newE[newlabel,j] = newE[newlabel,j]+2*changeN
                newE[oldlabel,j] = newE[oldlabel,j]-changeN+changeO
            elif j==oldlabel:
                newE[newlabel,j] = newE[newlabel,j]-changeN+changeO
                newE[oldlabel,j] = newE[oldlabel,j]-2*changeO
            else :
                change = sum(np.exp(np.dot(self.Z[indice,posi[j],:],self.gamma)))
                newE[newlabel,j] = newE[newlabel,j]+change
                newE[oldlabel,j] = newE[oldlabel,j]-change
        newE[:,newlabel] = np.transpose(newE[newlabel,:])
        newE[:,oldlabel] = np.transpose(newE[oldlabel,:])
        return newE

    def _updateNum(self,oldnum,oldcommunity,indice,newlabel):
        newnum=np.copy(oldnum)
        oldlabel = oldcommunity[indice]
        newnum[newlabel]=newnum[newlabel]+1
        newnum[oldlabel]=newnum[oldlabel]-1
        return newnum

    def fit(self,community_init = np.array([0]),gt = np.array([]), tabu_size=100, init = 30, max_iterations=5000, max_stay=500, children=2):
        
        if np.min(np.unique(community_init,return_counts=True)[1])>10:
            community_res = community_init
        else:
            community_res = np.random.randint(self.k, size=self.n)
        
        old_O = cf.O(community_res,self.Ab);old_E = cf.E(community_res,self.gamma,self.Z);old_num = cf.num(community_res)
        obj_res = cf.nLL(community_res,self.gamma,self.Ab,self.Z)
        
        init_cnt = 0
        while init_cnt < init:
            if np.min(np.unique(community_init,return_counts=True)[1])>10:
                community = community_init
            else:
                community = np.random.randint(self.k, size=self.n)
            old_O = cf.O(community,self.Ab);old_E = cf.E(community,self.gamma,self.Z);old_num = cf.num(community)
            obj = cf.nLL(community,self.gamma,self.Ab,self.Z)

            iteration = 0
            stay = 0
            cnt = 0
            tabu_set = np.array([],dtype=int)
            
            while (iteration<max_iterations) and (stay < max_stay):
            
                index = np.random.choice(np.setdiff1d(np.arange(self.n),tabu_set))
                
                tabu_set = np.append(index,tabu_set)
                tabu_set = tabu_set[:tabu_size]
                
            
                for label in np.setdiff1d(random.sample(range(0, self.k), children),community[index]):
                    new_O = self._updateO(old_O,community,index,label)
                    new_E = self._updateE(old_E,community,index,label)
                    new_num = self._updateNum(old_num,community,index,label)
                    
                    newnLL = (np.sum(new_O*np.log(new_E))/2-np.nansum(new_O*np.log(new_O)/2-new_num*np.log(new_num/self.n)))/(self.n**2)
                    if newnLL < obj:
                        stay = 0
                        old_O=new_O;old_E=new_E;old_num=new_num
                        community[index] = label
                        obj = newnLL
                        cnt+=1
                    else:
                        stay += 1
                            
                iteration += 1
            
            
            if(obj<obj_res):
                community_res = community
                obj_res = obj
            
            #print(init_cnt,iteration,stay,cnt)
            
            #if gt.shape[0]==community.shape[0]:
                #print(obj_res)
                #print('ARI is', adjusted_rand_score(community,gt))
                            
            

            init_cnt += 1
            
    

        return(community_res,obj_res)

