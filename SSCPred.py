from tensorflow.python import debug as tfdbg
import sys
#from progressbar import *
from scipy.linalg import toeplitz
import os
import matplotlib.pyplot as plt
import pickle
import subprocess
from itertools import  *
import numpy as np
import tensorflow as tf
import time
import random
import Nevaluation
from sklearn.cluster import KMeans
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dr=0
    #np.set_printoptions(threshold=np.nan)
aa=['A','R','D','C','Q','E','H','I','G','N','L','K','M','F','P','S','T','W','Y','V','-']
l2=tf.contrib.layers.l2_regularizer(0.05)
#l2=tf.contrib.layers.l2_regularizer(0.5)

def Converse_to_pairwise(tensor, l):

    t = tf.constant([1], tf.int32)
    o = tf.concat([tf.tile(tf.transpose(tensor, [0, 3, 1, 2]), tf.concat([t, l, t, t], axis=0)),
                   tf.tile(tf.transpose(tensor, [0, 1, 3, 2]), tf.concat([t, t, l, t], axis=0))], axis=3)
    return o

def residual_block(x, input_dim, kernel_n, kernel_size,tf_is_training):
    activation = tf.nn.elu
    x2 = tf.layers.conv2d(x,kernel_n,kernel_size,padding='same',activation=None,kernel_regularizer=l2,use_bias=False,kernel_initializer=tf.initializers.random_normal(stddev=0.01))
    x2 = activation(tf.layers.batch_normalization(x2, axis=3))
    #x2 = activation(x2)
    x2=tf.layers.conv2d(x2,kernel_n,kernel_size,padding='same',activation=None,kernel_regularizer=l2,use_bias=False,kernel_initializer=tf.initializers.random_normal(stddev=0.01))
    #x2 = activation(x2)
    #x2=tf.layers.dropout(x2,rate=dr,training=tf_is_training)
    x2 = tf.layers.batch_normalization(x2, axis=3)
    t=tf.add(x2,x)
    return t

def stage_1(X,ss,length,y,tf_is_training,threshold): 
    ss=tf.expand_dims(ss,axis=-1)
    #ss=tf.concat([do,ss],axis=2)
    ss=Converse_to_pairwise(ss,length)
    X=tf.squeeze(X)
    t=tf.one_hot(X,21)
    t1=tf.reshape(t,[-1,1])
    t2=tf.reshape(tf.transpose(t,[1,0]),[1,-1])
    cxk=tf.matmul(t1,t2)
    cxk=tf.reshape(cxk,[length[0],-1,length[0]])
    cxk=tf.transpose(cxk,[0,2,1])
    t=tf.expand_dims(cxk, 0)
    de=tf.reduce_sum(t,axis=-1)
    t=tf.concat([t,ss],axis=3)
    t.set_shape([None,None,None,487])

    t=tf.layers.conv2d(t,96,[3,3],padding='same',activation=tf.nn.elu,kernel_regularizer=l2,kernel_initializer=tf.initializers.random_normal(stddev=0.01))
    #t=tf.layers.dropout(t,rate=dr,training=tf_is_training)
    for i in range(30):
        t= residual_block(t, 96,96, [3, 3],tf_is_training)
    t=tf.layers.conv2d(t,1,[3,3],padding='same',activation=tf.nn.sigmoid,kernel_regularizer=l2,kernel_initializer=tf.initializers.random_normal(stddev=0.01))


    return t


def inference2(saved_model,seq_d,ss_d):
    seq_d=np.expand_dims(seq_d, axis=0)
    ss_d=np.expand_dims(ss_d, axis=0)
    #print(seq_d.shape)
    #print(ss_d.shape)
    T=8

    length=tf.placeholder(tf.int32,[None,])
    threshold=tf.placeholder(tf.float32)
    x= tf.placeholder(tf.int32, [1,None,])
    ss= tf.placeholder(tf.float32, [1,None,23])
    f_X= tf.placeholder(tf.int32, [None,])
    tf_is_training = tf.placeholder(tf.bool, None)
    y=None
    _y=stage_1(x,ss,length,y,tf_is_training,threshold)
    best=0.1
    saver2= tf.train.Saver()
    with tf.Session() as sess:
        for i in range(1):
            a_res=[0 for i in range(99)]
            step=0
            #fight=open(f'res/fight','wb') 
            #save_model=f'/data/chenmingcai/s2/model/ASSws'
            if saved_model is None:
                sess.run(tf.global_variables_initializer())
            else:
                #print('load model',saved_model)
                saver2.restore(sess,saved_model)
            prediction=sess.run(_y,feed_dict={tf_is_training:False,threshold:T,x:seq_d,ss:ss_d,length:[seq_d.shape[1]]})
            prediction=np.squeeze(prediction)
            l=prediction.shape[0]
            for i in range(l-1):
                for j in range(i,l):
                    print(i+1,j+1,prediction[i,j])

            #print(prediction)

if __name__=='__main__':
    parser.add_argument('--seq', default='sequence to be predict', type=str)
    parser.add_argument('--i1c', default='i1c files from SPIDER3\'s SPIDER3-Single_np', type=str)
    args = parser.parse_args()
    seq=args.seq #sys.argv[1]
    #with open(f'SPIDER3-Single_np/example_data/outputs/temp.i1c') as f:
    with open(args.i1c) as f:
        for j in f.readlines()[1:]:
            t=j[:-1].split(' ')
            t=[float(j) for j in t]
            d.append(t)
        d=np.asarray(d)

    seq_coded=list(map(lambda x:aa.index(x)+1 if x in aa else 0,seq))
    seq_coded=np.asarray(seq_coded)
    inference2(f'model/ASSwide3-27',seq_coded,d)
    exit()

