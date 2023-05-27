import glob
import numpy as np
import os
import argparse
import csv 

parser = argparse.ArgumentParser(description= 'Setting the Parameters.')
parser.add_argument('-dir', type=str, default="", help='folder') 
parser.add_argument('-rm', type=bool, default=False, help='remove log')
args = parser.parse_args()
files=glob.glob("../output/"+args.dir+"/*_"+args.dir+".log")

res = [[] for _ in range(10)]

for file in files:
    f=open (file, 'r').readlines () 
    for i in range(10):
        res[i].append(list (map(float, f[i].rstrip('\n').split())))
    if args.rm:
        os.remove(file)

with open("../output/"+args.dir+"/res.csv","w") as f:
    wr = csv.writer (f)
    wr.writerows(res)