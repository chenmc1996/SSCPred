import math
import gc
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import *
import pickle
def contact(i,j):
    return abs(i - j) >= 6
    
def short(i, j):
    return abs(i - j) <= 11 and abs(i - j) >= 6

def medium(i, j):
    return abs(i - j) <= 23 and abs(i - j) >= 12

def long(i, j):
    return abs(i - j) >= 24
    
def mask(length,r=contact):
    m=np.zeros((length,length),dtype=bool)
    for i in range(length):
        for j in range(length):
            if r(i,j):
                m[i][j]=True
    return m

def evaluation(native,prediction,L,K,range):#native contains -1 represent unknow
    num=int(L/K)
    prediction=np.where(mask(L),prediction,np.nan)
    bond=np.sort(prediction,axis=None)[num]
    if np.isnan(bond):
        bond=np.nanmax(prediction)
    prediction=np.where(prediction<=bond,prediction,np.nan)
    prediction=np.where(mask(L,range),prediction,np.nan)
    return np.count_nonzero(~np.isnan(prediction[np.logical_and(native<=8,native>=0)])),np.count_nonzero(~np.isnan(prediction[native!=-1]))

def evaluation_all(native,prediction,domain=None):
    if domain is not None:
        native=native.copy()
        prediction=prediction.copy()
    S=native.shape[0]
    M=np.zeros((S,S))
    D=[]
    if domain is not None:
        for i in domain:
            D=D+list(range(i[0]-1,i[1]))
        for i in range(S):
            for j in range(S):
                if not(i in D and j in D):
                    native[i,j]=-1
    count=0
    for i in range(S):
        if native[i,i]!=-1:
            count=count+1
        
    prediction=np.where(native!=-1,prediction,np.nan)
    K=[2]
    for i in K:
        LK=math.ceil(count/i)
        p=np.where(mask(S,long),prediction,np.nan)
        sorted_prediction=np.sort(p,axis=None)
        sorted_prediction=sorted_prediction[~np.isnan(sorted_prediction)]
        bond=sorted_prediction[LK]
        p=np.where(p<bond,p,np.nan)
        tT=np.count_nonzero(~np.isnan(p[np.logical_and(native<=8,native>=0)]))
        return tT/LK


def evaluation_all2(native,prediction,threshold=8,domain=None):
    native=native.copy()
    prediction=prediction.copy()
    prediction+=np.random.rand(*(prediction.shape))/10000000
    


    S=native.shape[0]
    for j in range(S):
        for k in range(j):
            prediction[j][k]=(prediction[j][k]+prediction[k][j])/2
            prediction[k][j]=0
    D=[]
    if domain is not None:
        for i in domain:
            D=D+list(range(i[0]-1,i[1]))
        for i in range(S):
            if not i in D:
                native[i,:]=-1
                native[:,i]=-1
    count=0
    for i in range(S):
        if native[i,i]!=-1:
            count=count+1
    prediction=np.where(native!=-1,prediction,np.nan)
    K=[5,2]
    res=[]
    t1=native
    t2=prediction
    for i in K:
        for ran in [long]:
        #for ran in [long]:
            #assert (native-t1).sum()==0
            #assert (prediction-t2)[~np.isnan(prediction-t2)].sum()==0
        #for ran in [long]:
            LK=math.ceil(count/i)
            #LK=math.ceil(5)
            p=np.where(mask(S,ran),prediction,np.nan)
            sorted_prediction=np.sort(p,axis=None)

            sorted_prediction=sorted_prediction[~np.isnan(sorted_prediction)]
            try:
                bond=sorted_prediction[-LK-1]
            except IndexError:
                #b=mask(S,long).sum()
                #a=np.logical_and(native<=threshold,native>=0)[mask(S,long)].sum()
                #print(S,a,b)
                res.append(0)
                continue
            #return p>bond,LK
            p=np.where(p>bond,p,np.nan)
            tT=np.count_nonzero(~np.isnan(p[np.logical_and(native<=threshold,native>=0)]))
            #print(tT)
            assert tT/LK<=1

            res.append(tT/LK)
    #print(res)
    return res

def evaluation_all3(native,prediction,K=2,domain=None):
    native=native.copy()
    prediction=prediction.copy()
    l=native.shape[0]
    M=np.zeros((l,l))
    D=[]
    if domain is not None:
        for i in domain:
            D=D+list(range(i[0]-1,i[1]))
        for i in range(l):
            if not i in D:
                native[i,:]=-1
                native[:,i]=-1
    count=0
    for i in range(l):
        if native[i,i]!=-1:
            count=count+1
    prediction=np.where(native!=-1,prediction,0)

    LK=math.ceil(count/K)
    #LK=math.ceil(5)
    p=np.where(mask(l,long),prediction,0)
    bond=np.partition(p,-LK,axis=None)[-LK]
    if bond==0:
        return 0
    p=np.where(p>=bond,p,0)
    return p
    tT=np.count_nonzero((p[np.logical_and(native<=8,native>=0)]))
    assert tT/LK<=1
    return p,tT/LK

def rr2prediction(path,seq_len):
    prediction=np.zeros((seq_len,seq_len))
    def g():
        with open(path) as f:
            for i in f.readlines():
                yield i
    line=g()
    t=next(line)
    #while True:
    #    t=next(line)
    #    if t[0].isdigit():
    #        break
    try:
        while True:
            ts=t.split(" ")
            prediction[int(ts[1])-1,int(ts[0])-1]=float(ts[-1][:-1])
            t=next(line)
    except StopIteration:
        pass
    prediction=prediction+prediction.T
    return prediction

if __name__ == '__main__':
    domain12={'T0859-D1': [[4, 132]], 'T0862-D1': [[139, 239]], 'T0863-D1': [[26, 218]], 'T0863-D2': [[219, 574]], 'T0864-D1': [[1, 246]], 'T0866-D1': [[38, 141]], 'T0869-D1': [[3, 106]], 'T0870-D1': [[2, 124]], 'T0886-D1': [[21, 44], [172, 216]], 'T0886-D2': [[45, 171]], 'T0892-D1': [[15, 83]], 'T0892-D2': [[84, 193]], 'T0896-D1': [[39, 124]], 'T0896-D2': [[125, 325]], 'T0896-D3': [[326, 486]], 'T0897-D1': [[24, 161]], 'T0897-D2': [[162, 285]], 'T0898-D1': [[4, 109]], 'T0898-D2': [[110, 164]], 'T0900-D1': [[5, 106]], 'T0904-D1': [[61, 311]], 'T0912-D1': [[24, 113], [299, 622]], 'T0912-D2': [[114, 154], [258, 299]], 'T0912-D3': [[155, 257]], 'T0918-D1': [[42, 149]], 'T0918-D2': [[150, 272]], 'T0918-D3': [[273, 411]], 'T0941-D1': [[124, 242], [247, 468]]}
    #with open('oneseq_test_100cut') as f:

    with open('res/hard_result2','rb') as f:
    #with open('res/result','rb') as f:
        native=pickle.load(f)
    with open('hard_set_dict') as f:
        dl=eval(f.readline())

    #with open('casp_set_dict') as f:
    #with open('res/cath1000_set') as f:
    #    dl=eval(f.readline())

    with open('res/casp_result','rb') as f:
    #with open('res/result','rb') as f:
        native=pickle.load(f)
    #for i in dl:
    #    if not os.path.exists(f'/data/chenmingcai/ResPRE/test/{i}.out'):
    #        assert False
    d_sum_r_r2=[]
    d_sum_r2_r3=[]
    d_sum_r_r3=[]
    l2={}
    dl=['T0761', 'T0763', 'T0767', 'T0771', 'T0777', 'T0781', 'T0785', 'T0789', 'T0790', 'T0791', 'T0794', 'T0806', 'T0808', 'T0810', 'T0814', 'T0820', 'T0824', 'T0827', 'T0831', 'T0832', 'T0834', 'T0836', 'T0837', 'T0855', 'T0859', 'T0860', 'T0861', 'T0862', 'T0863', 'T0864', 'T0865', 'T0866', 'T0867', 'T0868', 'T0869', 'T0870', 'T0871', 'T0872', 'T0873', 'T0878', 'T0879', 'T0880', 'T0886', 'T0889', 'T0891', 'T0892', 'T0893', 'T0896', 'T0897', 'T0898', 'T0900', 'T0902', 'T0903', 'T0904', 'T0911', 'T0912', 'T0918', 'T0920', 'T0921', 'T0922', 'T0928', 'T0941', 'T0942', 'T0943', 'T0944', 'T0945', 'T0947']
    dl=['T0761', 'T0763', 'T0771', 'T0777', 'T0781', 'T0785', 'T0820', 'T0832', 'T0859', 'T0860', 'T0862', 'T0867', 'T0880', 'T0896', 'T0897', 'T0898', 'T0900', 'T0904', 'T0928', 'T0941',]

    #dl=['T0880']
    print(len(dl))
    for i in dl:
        #print(i)
        l=native[i][0].shape[0]
        #with open(f'test_aln/{i}.out') as f:
        with open(f'Triresult/{i}.con') as f:
        ####with open(f'oneseq_aln/{i}.out') as f:
            r=np.zeros((l,l))
            for j in f.readlines():
                t=j.split()
                r[int(t[0])-1,int(t[1])-1]=float(t[2])
        #r=rr2prediction(f'test_aln/{i}.aln.con',native[i][0].shape[0])

        #t=evaluation_all2(native[i][0],native[i][1])
        t=evaluation_all2(native[i][0],r)
        if l>24 :
            l2[i]=t
            print(i,'%10.3f %10.3f'%(l2[i][0],l2[i][1]))
    print(l2)
    exit()
    #print(sum(d_sum_r_r2))
    #print(sum(d_sum_r2_r3))
    #print(sum(d_sum_r_r3))
    for j in range(1):
        print('%10.3f'%(sum([l2[i][j] for i in l2])/len(l2)))
    #print(sum([l2[i][0] for i in l2])/len(l2))
    #print(sum([l2[i][1] for i in l2])/len(l2))
    exit()
    for i in l2:
        print(i,'%10.2f'%(l2[i][0]*100),'%10.2f'%(l2[i][1]*100))
        


