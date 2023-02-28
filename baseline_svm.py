import torch,torchvision
import numpy as np
import torchvision.transforms as transforms
import scipy.io as sio
from sklearn import svm
from numpy import append, average

EEG_X=sio.loadmat('EEG_X.mat')
EEG_X_0=EEG_X['X'][0]
EEG_Y=sio.loadmat('EEG_Y.mat')
EEG_X=EEG_X['X']
EEG_Y=EEG_Y['Y']

scale_num=15

def data_processing():
    #print(EEG_X[0][1].shape)
    X=[]
    Y=[]
    for i in range(scale_num):
        index=list(range(scale_num))
        del index[i]
        d = np.vstack([EEG_X[0][j] for j in index])
        l = np.vstack([EEG_Y[0][j] for j in index])
        #print([EEG_Y[0][j] for j in index])
        l = np.reshape(l,(l.shape[0]))
        X.append(d)
        Y.append(l)
    eeg_x = [EEG_X[0][j] for j in range(scale_num)]
    #print(len(X[0]))
    eeg_y = [np.reshape(EEG_Y[0][j],(EEG_Y[0][j].shape[0])) for j in range(scale_num)]
    return X, Y,eeg_x, eeg_y


def output_precision(vector_classify,c,d):
    n=c.shape[0]
    predict_res=vector_classify.predict(c)
    predict_cor=np.sum(predict_res==d)
    rate=predict_cor/n
    print(predict_res,n,rate)
    return rate

def self_svm(X,Y,eeg_x,eeg_y):
    precision=[]
    for i in range(scale_num):
        x=X[i]
        y=Y[i]
        clf=svm.SVC(decision_function_shape='ovo')
        clf.fit(x,y)
        p=output_precision(clf,eeg_x[i],eeg_y[i])
        precision.append(p)
        print(f"precision of current data set(set{i}) is {precision[i]}")
    
    print(f"total average precision:{average(precision)}")


if __name__=="__main__":
    x,y,eeg_x,eeg_y=data_processing()
    self_svm(x,y,eeg_x,eeg_y)

    
