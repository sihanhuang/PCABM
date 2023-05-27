import glob
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description= 'e4etting the Parameters.')
parser.add_argument('-dir', type=str, default="", help='folder') 
parser.add_argument('-rm', type=bool, default=False, help='remove log')
args = parser.parse_args()
files=glob.glob("../output/"+args.dir+"/*_"+args.dir+".log")

k1 = []
k2 = []
k3 = []

for file in files:
    f=open (file, 'r'). readlines()
    k1 += list(map(float, f[0].rstrip('\n').split())) 
    k2 += list(map(float, f[1].rstrip('\n').split()))
    k3 += list(map(float, f[2].rstrip('\n').split())) 
    if args.rm:
        os.remove(file)

k1 = np.array(k1).reshape(len(files), len(k1)//len(files))
k2 = np.array(k2).reshape(len(files), len(k2)//len(files))
k3 = np.array(k3).reshape(len(files), len(k3)//len(files))

np.savetxt("../output/"+args.dir+"/k1.csv", k1, delimiter=",")
np.savetxt("../output/"+args.dir+"/k2.csv", k2, delimiter=",")
np.savetxt("../output/"+args.dir+"/k3.csv", k3, delimiter=",") 