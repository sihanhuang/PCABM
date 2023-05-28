import glob
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description= 'e4etting the Parameters.')
parser.add_argument('-dir', type=str, default="", help='folder') 
parser.add_argument('-rm', type=bool, default=False, help='remove log')
args = parser.parse_args()
files=glob.glob("../output/"+args.dir+"/*_"+args.dir+".log")

e1 = []
e2 = []

for file in files:
    f=open (file, 'r'). readlines()
    e1 += list(map(float, f[0].rstrip('\n').split())) 
    e2 += list(map(float, f[1].rstrip('\n').split()))
    if args.rm:
        os.remove(file)

e1 = np.array(e1).reshape(len(files), len(e1)//len(files))
e2 = np.array(e2).reshape(len(files), len(e2)//len(files))

np.savetxt("../output/"+args.dir+"/e1.csv", e1, delimiter=",")
np.savetxt("../output/"+args.dir+"/e2.csv", e2, delimiter=",")