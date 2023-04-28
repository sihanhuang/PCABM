 import numpy as np
 import pandas as pd
 import random
 import pcabm.commFunc as cf
 from sklearn.metrics import normalized_mutual_info_score
 from sklearn.metrics.cluster import adjusted_rand_score


 # Community Related Quantities Initialization

 class DCBM():

     def __init__(self, Ab, k):
         self.Ab = Ab
         self.k = k
         self.n = Ab.shape[0]

     # Likelihood Update
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

     def fit(self,community_init, gt = np.array([0]), tabu_size=100, init = 30, max_iterations=5000, max_stay=500, children=2):

         if np.min(np.unique(community_init,return_counts=True)[1])>10:
             community_res = community_init
         else:
             community_res = np.random.randint(self.k, size=self.n)

         old_O = cf.O(community_init,self.Ab);
         obj_res = -np.sum(old_O*np.log(old_O/cf.summatrix(old_O)))

         init_cnt = 0;
         while init_cnt<init:
             if np.min(np.unique(community_init,return_counts=True)[1])>10:
                 community = community_init
             else:
                 community = np.random.randint(self.k, size=self.n)

             old_O = cf.O(community,self.Ab);
             obj = -np.sum(old_O*np.log(old_O/cf.summatrix(old_O)))


             tabu_set = np.array([],dtype=int)
             iteration = 0
             stay = 0
             while (iteration < max_iterations) and (stay < max_stay):
                 index =  random.randint(0,self.n-1) # Generate one randomly
                 while index in tabu_set:
                     index =  random.randint(0,self.n-1) # Generate another
                 tabu_set = [index] + tabu_set[:tabu_size-1]
                 stay = stay+1
                 for label in np.setdiff1d(random.sample(range(0, self.k), children),community[index]):
                     new_O = self._updateO(old_O,community,index,label)
                     newnLL = -np.sum(new_O*np.log(new_O/cf.summatrix(new_O)))
                     if newnLL < obj:
                         stay = 0
                         old_O=new_O
                         community[index] = label
                         obj = newnLL
                 iteration = iteration + 1

             init_cnt += 1

             if(obj<obj_res):
                 community_res = community
                 obj_res = obj

         return(community_res,obj_res)