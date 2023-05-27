import glob
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description= 'Setting the Parameters.')
parser.add_argument('-dir', type=str, default="", help='folder') 
parser.add_argument('-rm', type=bool, default=False, help='remove log')
args = parser.parse_args()
files=glob.glob("../output/"+args.dir+"/*_"+args.dir+".log")

T = []
F = []
B = []
S = []
N = []

for file in files:
    f=open (file, 'r'). readlines()
    T += list(map(float, f[0].rstrip('\n').split())) 
    F += list(map(float, f[1].rstrip('\n').split()))
    B += list(map(float, f[2].rstrip('\n').split())) 
    S += list(map(float, f[3].rstrip('\n').split())) 
    N += list(map(float, f[4].rstrip('\n').split())) 
    if args.rm:
        os.remove(file)

T = np.array(T).reshape(len(files), len(T)//len(files))
F = np.array(F).reshape(len(files), len(F)//len(files))
B = np.array(B).reshape(len(files), len(B)//len(files))
S = np.array(S).reshape(len(files), len(S)//len(files))
N = np.array(N).reshape(len(files), len(N)//len(files))

np.savetxt("../output/"+args.dir+"/T.csv", T, delimiter=",")
np.savetxt("../output/"+args.dir+"/F.csv", F, delimiter=",")
np.savetxt("../output/"+args.dir+"/B.csv", B, delimiter=",") 
np.savetxt("../output/"+args.dir+"/S.csv", S, delimiter=",")
np.savetxt("../output/"+args.dir+"/N.csv", N, delimiter=",")