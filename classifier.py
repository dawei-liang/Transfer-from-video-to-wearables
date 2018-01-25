# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 13:03:44 2017

@author: david
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
import sklearn.preprocessing as sklprep
from itertools import islice
import itertools as it
import scipy.stats as spstas

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

#Load IMU data
data1 = pd.read_csv('G:/source_code/wearables_data/drinking1.csv')   #Drinking
data2 = pd.read_csv('G:/source_code/wearables_data/drinking2.csv')
data3 = pd.read_csv('G:/source_code/wearables_data/drinking3.csv')
data4 = pd.read_csv('G:/source_code/wearables_data/waving1.csv')   #Waving
data5 = pd.read_csv('G:/source_code/wearables_data/waving2.csv')
data6 = pd.read_csv('G:/source_code/wearables_data/waving3.csv')
data7 = pd.read_csv('G:/source_code/wearables_data/drinking_waving1.csv')   #Mix
data8 = pd.read_csv('G:/source_code/wearables_data/drinking_waving2.csv')
data9 = pd.read_csv('G:/source_code/wearables_data/drinking_waving3.csv')

#Load video data
vid1 = pd.read_csv('G:/source_code/video_data/drinking1.csv')
vid2 = pd.read_csv('G:/source_code/video_data/drinking2.csv')
vid3 = pd.read_csv('G:/source_code/video_data/drinking3.csv')
vid4 = pd.read_csv('G:/source_code/video_data/waving1.csv')
vid5 = pd.read_csv('G:/source_code/video_data/waving2.csv')
vid6 = pd.read_csv('G:/source_code/video_data/waving3.csv')
vid7 = pd.read_csv('G:/source_code/video_data/drinking_waving1.csv') 
vid8 = pd.read_csv('G:/source_code/video_data/drinking_waving2.csv')
vid9 = pd.read_csv('G:/source_code/video_data/drinking_waving3.csv')

#Downsample to 10Hz
def downsampling(x):
    dd = sp.resample(x, int(x.shape[0] / 10))
    return dd

        # ----------------------------- Segmentaion -------------------------------------

#Activity offsets
#Training data: accx,y
#Drinking 1
drinking1 = [18,66,110,157,209,257,305,351,394,444,492,539,585,634,676,721,770,810,856,898]
#Drinking 2
drinking2 = [8,57,105,157,211,257,306,356,399,449,493,536,589,631,670,711,751,791,835,871]
#Drinking 3
drinking3 = [6,52,108,159,212,260,314,366,419,470,518,560,604,654,698,736,780,827,866,899]
#Waving 1
waving1 = [4,71,143,211,292,372,446,522,599]
#Waving 2
waving2 = []
#Waving 3
waving3 = []

#Test data: xv,yv
drinking_video1 = [2,52,103,158,211,259,313,362,409,462,508,562,613,668,714,754,810,864,918,960]
waving_video1 = []

        # ----------------------------- Define functions -------------------------------------

#Scale data to range 0~1
def scale(x):    
    min_max_scaler = sklprep.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(x)
    return X_train_minmax

#Sliding windows (fixed size)    
def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step*length) for stream, i in zip(streams, it.count(step=step))])

def segment(x, y, activity):
    feature = list()  
    feature_vec=list()
    label = list()
    
    if activity == drinking1 or activity == drinking2 or activity == drinking3:   
        #Filter each activity segment
        for i in range (len(drinking1)-1):
            temp1 = list()   
            temp2 = list()
            #Assume single activity length = 40
            for j in range (0,40):
                temp1.append(x[[activity[i]+j], 0])
                temp2.append(y[[activity[i]+j], 0])

            MX = np.mean(temp1)
            VX = np.var(temp1)
            SX = spstas.skew(temp1)
            KX = spstas.kurtosis(temp1)
            RX = np.sqrt(np.mean(temp1)) 
            
            MY = np.mean(temp2)
            VY = np.var(temp2)
            SY = spstas.skew(temp2)
            KY = spstas.kurtosis(temp2)
            RY = np.sqrt(np.mean(temp2)) 
            
            feature = np.hstack((MX,VX))   #feature_i = [MX,VX,SX,KX,RX]
            feature = np.hstack((feature,SX))
            feature = np.hstack((feature,KX))
            feature = np.hstack((feature,RX))
            
            feature = np.hstack((feature,MY))   #feature_i = [MX,VX,SX,KX,RX,MY,VY,SY,KY,RY]
            feature = np.hstack((feature,VY)) 
            feature = np.hstack((feature,SY))
            feature = np.hstack((feature,KY))
            feature = np.hstack((feature,RY))
            feature_vec.append(feature)   #feature_vec = [feature_1, feature_2, ...]
            
            label.append('drinking')   #Label = [label1, label2, ...]
        return feature_vec, label
    
    
    elif activity == waving1:   
        #Filter each activity segment
        for i in range (len(waving1)):
            temp1 = list()   
            temp2 = list()
            #Assume single activity length = 50
            for j in range (0,50):
                temp1.append(x[[activity[i]+j], 0])
                temp2.append(y[[activity[i]+j], 0])

            MX = np.mean(temp1)
            VX = np.var(temp1)
            SX = spstas.skew(temp1)
            KX = spstas.kurtosis(temp1)
            RX = np.sqrt(np.mean(temp1)) 
            
            MY = np.mean(temp2)
            VY = np.var(temp2)
            SY = spstas.skew(temp2)
            KY = spstas.kurtosis(temp2)
            RY = np.sqrt(np.mean(temp2)) 
            
            feature = np.hstack((MX,VX))   #feature_i = [MX,VX,SX,KX,RX]
            feature = np.hstack((feature,SX))
            feature = np.hstack((feature,KX))
            feature = np.hstack((feature,RX))
            
            feature = np.hstack((feature,MY))   #feature_i = [MX,VX,SX,KX,RX,MY,VY,SY,KY,RY]
            feature = np.hstack((feature,VY)) 
            feature = np.hstack((feature,SY))
            feature = np.hstack((feature,KY))
            feature = np.hstack((feature,RY))
            feature_vec.append(feature)   #feature_vec = [feature_1, feature_2, ...]
            
            label.append('waving')   #Label = [label1, label2, ...]
        return feature_vec, label

def testdata_segment(x,y, activity):
    feature = list()  
    feature_vec=list()
    
    if activity == "drinking1":   
        #Filter each activity segment
        for i in range (len(drinking1)-1):
            temp1 = list()   
            temp2 = list()
            #Assume single activity length = 42
            for j in range (0,42):
                temp1.append(x[[drinking1[i]+j], 0])
                temp2.append(y[[drinking1[i]+j], 0])

            MX = np.mean(temp1)
            VX = np.var(temp1)
            SX = spstas.skew(temp1)
            KX = spstas.kurtosis(temp1)
            RX = np.sqrt(np.mean(temp1)) 
            
            MY = np.mean(temp2)
            VY = np.var(temp2)
            SY = spstas.skew(temp2)
            KY = spstas.kurtosis(temp2)
            RY = np.sqrt(np.mean(temp2)) 
            
            feature = np.hstack((MX,VX))   #feature_i = [MX,VX,SX,KX,RX]
            feature = np.hstack((feature,SX))
            feature = np.hstack((feature,KX))
            feature = np.hstack((feature,RX))
            
            feature = np.hstack((feature,MY))   #feature_i = [MX,VX,SX,KX,RX,MY,VY,SY,KY,RY]
            feature = np.hstack((feature,VY)) 
            feature = np.hstack((feature,SY))
            feature = np.hstack((feature,KY))
            feature = np.hstack((feature,RY))
            feature_vec.append(feature)   #feature_vec = [feature_1, feature_2, ...]
            
        return feature_vec


def SVM(traindata, label, testdata):
    clf = LinearSVC(random_state=0)
    clf.fit(traindata, label)

    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
              verbose=0)

    print(clf.coef_)
    print(clf.intercept_)
    print(clf.predict(testdata))

        # ----------------------------- Segmentation & labeling -------------------------------------
        

#Segment drinking1 data and get labels
accx1 = data1[['xa']].values   
accx1 = downsampling(accx1)
accx1 = scale(accx1)
accy1 = data1[['ya']].values
accy1 = downsampling(accy1) 
accy1 = scale(accy1)

segmented_acc1, label_acc1 = segment(accx1, accy1, activity = drinking1)
#accx1 = list(moving_window(accx1, 25))

accx2 = data2[['xa']].values   
accx2 = downsampling(accx2) 
accx2 = scale(accx2)
accy2 = data2[['ya']].values
accy2 = downsampling(accy2) 
accy2 = scale(accy2)
segmented_acc2, label_acc2 = segment(accx2, accy2, activity = drinking2)

accx3 = data3[['xa']].values
accx3 = downsampling(accx3)
accx3 = scale(accx3)
accy3 = data3[['ya']].values
accy3 = downsampling(accy3) 
accy3 = scale(accy3)
segmented_acc3, label_acc3 = segment(accx3, accy3, activity = drinking3)

#Segment waving1 data and get labels
accx4 = data4[['xa']].values
accx4 = downsampling(accx4) 
accx4 = scale(accx4)

accy4 = data4[['ya']].values
accy4 = downsampling(accy4)
accy4 = scale(accy4)
segmented_acc4, label_acc4 = segment(accx4, accy4, activity = waving1)

        # ----------------------------- Train & Test SVM Model -------------------------------------
#Generate overall training data & labels        
training_data = segmented_acc1
training_data = np.vstack((segmented_acc1,segmented_acc2))
training_data = np.vstack((training_data,segmented_acc3))
training_data = np.vstack((training_data,segmented_acc4))

training_label = label_acc1
training_label = np.hstack((label_acc1,label_acc2))
training_label = np.hstack((training_label,label_acc3))
training_label = np.hstack((training_label,label_acc4))

#Generate test video data
xv1 = vid1[['x']].values 
xv1 = scale(xv1)

yv1 = vid1[['y']].values
yv1 = scale(yv1)
segmented_vid1 = testdata_segment(xv1, yv1, activity = "drinking1")

#Fit SVM model
SVM(training_data, training_label, segmented_vid1)

        # ----------------------------- Backup -------------------------------------

#Data Preprocessing: accx


accx5 = data5[['xa']].values
accx5 = downsampling(accx5)
accx5 = scale(accx5)
accx6 = data6[['xa']].values
accx6 = downsampling(accx6)
accx6 = scale(accx6) 
accx7 = data7[['xa']].values
accx7 = downsampling(accx7) 
accx7 = scale(accx7)
accx8 = data8[['xa']].values
accx8 = downsampling(accx8) 
accx8 = scale(accx8)
accx9 = data9[['xa']].values
accx9 = downsampling(accx9) 
accx9 = scale(accx9)

#Data Preprocessing: xv

xv2 = vid2[['x']].values
xv2 = scale(xv2) 
xv3 = vid3[['x']].values 
xv3 = scale(xv3)
xv4 = vid4[['x']].values
xv4 = scale(xv4) 
xv5 = vid5[['x']].values 
xv5 = scale(xv5)
xv6 = vid6[['x']].values 
xv6 = scale(xv6)
xv7 = vid7[['x']].values 
xv7 = scale(xv7)
xv8 = vid8[['x']].values 
xv8 = scale(xv8)
xv9 = vid9[['x']].values 
xv9 = scale(xv9)

#Data Preprocessing: accy


accy5 = data5[['ya']].values
accy5 = downsampling(accy5) 
accy5 = scale(accy5)
accy6 = data6[['ya']].values
accy6 = downsampling(accy6) 
accy6 = scale(accy6)
accy7 = data7[['ya']].values
accy7 = downsampling(accy7) 
accy7 = scale(accy7)
accy8 = data8[['ya']].values
accy8 = downsampling(accy8)
accy8 = scale(accy8)
accy9 = data9[['ya']].values
accy9 = downsampling(accy9)
accy9 = scale(accy9)

#Data Preprocessing: yv

yv2 = vid2[['y']].values 
yv2 = scale(yv2)
yv3 = vid3[['y']].values 
yv3 = scale(yv3)
yv4 = vid4[['y']].values 
yv4 = scale(yv4)
yv5 = vid5[['y']].values 
yv5 = scale(yv5)
yv6 = vid6[['y']].values
yv6 = scale(yv6)
yv7 = vid7[['y']].values 
yv7 = scale(yv7)
yv8 = vid8[['y']].values 
yv8 = scale(yv8)
yv9 = vid9[['y']].values 
yv9 = scale(yv9)

#Plot data
#plt.figure(1)
#plt.title('drinking_acc')
#plt.plot(accx1[:,0], 'r')
#plt.plot(accy1[:,0], 'g')
##plt.plot(accx2[:,0], 'r')
##plt.plot(accy2[:,0], 'g')
##plt.plot(accx3[:,0], 'r')
##plt.plot(accy3[:,0], 'g')
#plt.legend(['accx', 'accy'], loc='best')
#plt.plot(accz1[:,0])

#Plot video data x
#plt.figure(2)
#plt.title('drinking_vid')
#plt.plot(xv1[:,0], 'r')
#plt.plot(yv1[:,0], 'g')
##plt.plot(xv2[:,0], 'r')
##plt.plot(yv2[:,0], 'g')
##plt.plot(xv3[:,0], 'r')
##plt.plot(yv3[:,0], 'g')
#plt.legend(['xv', 'yv'], loc='best')
#
#plt.figure(3)
#plt.title('waving_acc')
#plt.plot(accx4[:,0], 'r')
#plt.plot(accy4[:,0], 'g')
##plt.plot(accx5[:,0], 'r')
##plt.plot(accy5[:,0], 'g')
##plt.plot(accx6[:,0], 'r')
##plt.plot(accy6[:,0], 'g')
##plt.plot(accz4[:,0])
#plt.legend(['accx', 'accy'], loc='best')
#
#plt.figure(4)
#plt.title('waving_vid')
#plt.plot(xv4[:,0], 'r')
#plt.plot(yv4[:,0], 'g')
##plt.plot(xv5[:,0], 'r')
##plt.plot(yv5[:,0], 'g')
##plt.plot(xv6[:,0], 'r')
##plt.plot(yv6[:,0], 'g')
#plt.legend(['xv', 'yv'], loc='best')
#
#plt.figure(5)
#plt.title('mix_acc')
#plt.plot(accx7[:,0], 'r')
#plt.plot(accy7[:,0], 'g')
##plt.plot(accx8[:,0], 'r')
##plt.plot(accy8[:,0], 'g')
##plt.plot(accx9[:,0], 'r')
##plt.plot(accy9[:,0], 'g')
#plt.legend(['accx', 'accy'], loc='best')
#
#plt.figure(6)
#plt.title('mix_vid')
##plt.plot(xv7[:,0], 'r')
##plt.plot(yv7[:,0], 'g')
##plt.plot(xv8[:,0], 'r')
##plt.plot(yv8[:,0], 'g')
#plt.plot(xv9[:,0], 'r')
#plt.plot(yv9[:,0], 'g')
#plt.legend(['xv', 'yv'], loc='best')
#
#plt.show()
