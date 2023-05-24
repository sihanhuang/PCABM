import numpy as np
import random
import pcabm.commFunc as cf
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


class SBM():

    def __init__(self, Ab, k):
        self.Ab = Ab
        self.k = k
        self.n = Ab.shape[0]

    def nLL_label(self,e):
        num_res = cf.num(e)
        O_res = cf.O(e,self.Ab)
        E_res = np.matmul(num_res.reshape(self.k,1),num_res.reshape(1,self.k))-np.diag(num_res)
        return (np.sum(O_res*np.log(E_res))/2-np.nansum(O_res*np.log(np.clip(O_res,a_min = 1,a_max=1e300))/2-num_res*np.log(num_res/self.n)))/(self.n**2)
        
    def updateO(self,oldO,oldcommunity,indice,newlabel):
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

    def updatenum(self,oldnum,oldcommunity,indice,newlabel):
        newnum=np.copy(oldnum)
        oldlabel = oldcommunity[indice]
        newnum[newlabel]=newnum[newlabel]+1
        newnum[oldlabel]=newnum[oldlabel]-1
        return newnum

    def tabu_search(self, ini_community, tabu_size=30, max_iterations=1000, max_stay=1000, children=2):
        community=np.copy(ini_community)
        old_O = cf.O(community,self.Ab);
        old_num = cf.num(community)
        obj = self.nLL_label(community)
        tabu_set = []
        iteration = 0
        stay = 0

        while (iteration < max_iterations) and (stay < max_stay):  # Stopping Criteria
            index =  random.randint(0,self.n-1) # Generate one randomly
            while index in tabu_set:
                index =  random.randint(0,self.n-1) # Generate another
            tabu_set = [index] + tabu_set[:tabu_size-1]
            stay = stay+1
            for label in np.setdiff1d(random.sample(range(0, self.k), children),community[index]):
                new_O = self.updateO(old_O,community,index,label)
                new_num = self.updatenum(old_num,community,index,label)
                new_E = np.matmul(new_num.reshape(self.k,1),new_num.reshape(1,self.k))-np.diag(new_num)
                if np.min(new_O)==0 or np.min(new_E)==0:
                    break
                newnLL = (np.sum(new_O*np.log(new_E))/2-np.nansum(new_O*np.log(new_O)/2-new_num*np.log(new_num/self.n)))/(self.n**2)
                if newnLL < obj:
                    stay = 0
                    old_O=new_O;old_num=new_num
                    community[index] = label
                    obj = newnLL

            iteration = iteration + 1
        #print(iteration,stay)
        return(community,obj)

    def fit(self,community_init = np.array([0]),gt = np.array([]), tabu_size=100, init = 30, max_iterations=1000, max_stay=500, children=2):
        
        obj_res = self.nLL_label(np.random.randint(self.k, size=self.n))

        init_cnt = 0
        while init_cnt < init:

            if np.min(np.unique(community_init,return_counts=True)[1])>10:
                community_res = community_init
            else:
                community_res = np.random.randint(self.k, size=self.n)

            community, obj = self.tabu_search(community_res, tabu_size , max_iterations, max_stay, children)
            
            if(obj<obj_res):
                community_res = community
                obj_res = obj
                print(obj)
            
            init_cnt += 1
            
        return(community_res,obj_res)

