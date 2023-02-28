import numpy as np
import torch
import scipy.io as sio
import scipy.linalg
import numpy as np
import scipy.io
import scipy.linalg
from sklearn import svm
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

scale_num = 1

txt_xs = np.loadtxt('./x314-3.txt')
txt_xt = np.loadtxt('./x314-14.txt')


def kernel(X1, X2, gamma):
    K = None
    if X2 is not None:
        K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
    else:
        K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)

    return K


class TCA:
    def __init__(self, dim=30, lamb=1, gamma=1):
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        :param Xs: 数据数 * 每个数据的特征数，源特征
        :param Xt: 数据数 * 每个数据的特征数，目标特征
        :return: 二者经过TCA之后的Xs_new和Xt_new
        '''

        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(X, None, gamma=self.gamma)
        n_eye = n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        print('start w/V 计算')
        w, V = scipy.linalg.eig(a, b, overwrite_a=True, overwrite_b=True, check_finite=False)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        Xs_new = np.dot(Xs_new, np.eye(Xs_new.shape[1])).astype(float)
        Xt_new = np.dot(Xt_new, np.eye(Xt_new.shape[1])).astype(float)
        print('TCA完成')
        return Xs_new, Xt_new


def data_processing(EEG_X, EEG_Y):
    X = []
    Y = []
    for i in range(scale_num):
        index = list(range(scale_num))
        # del index[i]
        d = np.vstack([EEG_X[j] for j in index])
        # l = np.vstack([EEG_Y[j] for j in index])
        # print([EEG_Y[0][j] for j in index])
        # l = np.reshape(l, (l.shape[0]))
        X.append(d)
        # Y.append(l)
    eeg_x = [EEG_X[j] for j in range(scale_num)]
    # print(len(X[0]))
    eeg_y = [np.reshape(EEG_Y, (EEG_Y.shape[0]))]
    print(len(X[0]), len(Y[0]), len(eeg_x), len(eeg_x))
    return X, Y, eeg_x, eeg_y


def output_precision(vector_classify, c, d):
    n = c.shape[0]
    predict_res = vector_classify.predict(c)
    np.savetxt('outputres.txt', predict_res)
    predict_cor = np.sum(predict_res == d)
    rate = predict_cor / n
    print(predict_res, n, rate)
    return rate


def transferred_svm(X_source, Y_source, X_target, Y_target):
    x = X_source
    y = Y_source
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x, y)
    p = output_precision(clf, X_target, Y_target)
    print(f"precision of current data set(set) is {p}")


if __name__ == '__main__':
    EEG_X = sio.loadmat('EEG_X.mat')['X'][0]
    EEG_Y = sio.loadmat('EEG_Y.mat')['Y'][0]

    Xs_group = []
    for i in range(14):
        Xs_group.append(EEG_X[i])
    Xs = EEG_X[2]

    # print(Xs.shape)
    #print(Xs_group[0].shape)
    Xt = EEG_X[14]

    #Xsnew_group = []

    #= for i in range(scale_num):
    #    tca = TCA(dim=30, lamb=1, gamma=1)
    #    Xs_new, Xt_new = tca.fit(Xs_group[i], Xt)
    #    Xsnew = np.array([*Xs_new])
    #    Xsnew_group.append(Xsnew)
    # dummy=np.ones((3394,30))
    # Xsnew_group.append(dummy)

    tca = TCA(dim=30, lamb=1, gamma=1)
    Xs_new, Xt_new = tca.fit(Xs, Xt)
    Xsnew = np.array([*Xs_new])
    Xtnew = np.array([*Xt_new])

    print('Xs/Xt size:', Xsnew.shape, Xtnew.shape)

    np.savetxt('x214-2.txt', Xsnew)
    np.savetxt('x214-14.txt', Xtnew)
    print('save over')

    # print(Xsnew_group[0].shape)
    # print(Xsnew_group[0].size)
    #transferred_X = np.asarray(Xsnew_group)
    #refer_Y = np.asarray(Xt)
    #print(transferred_X.shape)

    print('开始训练svm')
    # data_processing(transferred_X, refer_Y)
    np.savetxt('y_source.txt', EEG_Y[3].ravel())
    np.savetxt('y_target.txt', EEG_Y[14].ravel())
    transferred_svm(txt_xs, EEG_Y[3].ravel(), txt_xt, EEG_Y[14].ravel())

