import glob
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description= 'Setting the Parameters.')
parser.add_argument('-dir', type=str, default="", help='folder') 
parser.add_argument('-rm', type=bool, default=False, help='remove log')
args = parser.parse_args()
files=glob.glob("../output/"+args.dir+"/*_"+args.dir+".log")

PCA = []
SCWA = []

for file in files:
    f=open (file, 'r'). readlines()
    SCWA += list(map(float, f[0].rstrip('\n').split()))
    PCA += list(map(float, f[1].rstrip('\n') .split())) 
    if args.rm:
        os.remove(file)

SCWA = np.array(SCWA).reshape(len (files), len (SCWA) //len(files))
PCA = np.array (PCA).reshape(len (files), len (PCA) //len (files))

np.savetxt("../output/"+args.dir+"/SCWA.csv", SCWA, delimiter=",")
np.savetxt("../output/"+args.dir+"/PCA.csv",PCA, delimiter=",")