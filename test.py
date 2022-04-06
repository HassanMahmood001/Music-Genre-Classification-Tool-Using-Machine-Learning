from dis import dis
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import random 
import operator

import math
import numpy as np
from collections import defaultdict

dataset = []
def loadDataset(filename):
    with open("extracted_features.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

loadDataset("extracted_features.dat")

def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

def getNeighbors(trainingSet , instance , k):
    distances =[]
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors  

def nearestClass(neighbors):
    classVote ={}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1 
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]


results=defaultdict(int)

i=1
for folder in os.listdir("./genres/"):
    results[i]=folder
    i+=1
# print("\n",results,"\n")
directory="./wav_test/"
pd=[]
pdd=[]
res={}
for filename in os.listdir(directory):
    ao=os.path.join(directory, filename)
    (rate,sig)=wav.read(ao)
    mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
    covariance = np.cov(np.matrix.transpose(mfcc_feat))
    mean_matrix = mfcc_feat.mean(0)
    feature=(mean_matrix,covariance,0)

    pred=nearestClass(getNeighbors(dataset ,feature , 5))
    pd.append(filename)
    print(results[pred])
    
    # pdd.append(results[pred])
    if results[pred]=="blues":
        result="blues"
    elif results[pred]=="classical":
        result="classical"
    elif results[pred]=="country":
        result="country"
    elif results[pred]=="disco":
        result="disco"
    elif results[pred]=="hiphop":
        result="hiphop"
    elif results[pred]=="jazz":
        result="jazz"
    elif results[pred]=="metal":
        result="metal"
    elif results[pred]=="pop":
        result="pop"
    elif results[pred]=="reggae":
        result="reggae"
    else:
        result="rock"
                
    pdd.append(result)
    
    zip_iterator = zip(pd, pdd)
    res = dict(zip_iterator)
    print(res)

    


