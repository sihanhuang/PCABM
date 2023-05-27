import glob
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description= 'Setting the Parameters.')
parser.add_argument('-dir', type=str, default="", help='folder') 
parser.add_argument('-rm', type=bool, default=False, help='remove log')
args = parser.parse_args()
files=glob.glob("../output/"+args.dir+"/*_"+args.dir+".log")

SC = []
SCWA = []
SBM = []
PCA = []
DCBM = []
SCORE = []

for file in files:
    f=open (file, 'r'). readlines()
    SC += list(map(float, f[0].rstrip('\n').split())) 
    SCWA += list(map(float, f[1].rstrip('\n').split()))
    SBM += list(map(float, f[2].rstrip('\n').split())) 
    PCA += list(map(float, f[3].rstrip('\n') .split())) 
    DCBM += list(map(float, f[4].rstrip('\n').split())) 
    SCORE += list(map(float, f[5].rstrip('\n').split())) 
    if args.rm:
        os.remove(file)

SC = np.array(SC).reshape(len (files), len(SC) //len (files))
SCWA = np.array(SCWA).reshape(len (files), len (SCWA) //len(files))
SBM = np.array(SBM).reshape(len(files), len(SBM)//len(files))
PCA = np.array (PCA).reshape(len (files), len (PCA) //len (files))
DCBM = np.array (DCBM).reshape (len (files), len(DCBM) //len (files))
SCORE = np.array (SCORE).reshape (len (files), len(SCORE) //len (files))

np.savetxt("../output/"+args.dir+"/SC.csv", SC, delimiter=",")
np.savetxt("../output/"+args.dir+"/SCWA.csv", SCWA, delimiter=",")
np.savetxt("../output/"+args.dir+"/SBM.csv", SBM, delimiter=",") 
np.savetxt("../output/"+args.dir+"/PCA.csv",PCA, delimiter=",")
np.savetxt("../output/"+args.dir+"/DCBM.csv", DCBM, delimiter=",")
np.savetxt("../output/"+args.dir+"/SCORE.csv", SCORE, delimiter=",")