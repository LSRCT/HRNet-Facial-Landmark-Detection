import sklearn
from scipy.ndimage import uniform_filter1d
import numpy as np
import sys, os
import ast

def loadXY(path):
    X = []
    Y = []
    with open(path, "r") as f:
        # first one is header
        f.readline()
        # other ones are data
        for line in f:
            ls = line.strip("\n")
            ls = ls.replace(".", "")
            ls = ls.replace("[", "")
            ls = ls.replace("]", "")
            ls = ls.split(";")[1:]
            label= int(ls[-1])
            ls = [[int(coord[:3].strip(" ")), int(coord[3:].strip(" "))] for coord in ls[:-1]]
            X.append(ls)
            Y.append(label)
    return np.array(X), np.array(Y)


def preprocessXY(x,y):
    new_x = []
    for frame in x:
        f_new = np.linalg.norm(frame, axis=1)
        # center the face basically
        f_new = f_new - np.mean(f_new)
        new_x.append(f_new)
    # Lowpass filter to get rid of wiggling
    new_x = uniform_filter1d(new_x, 10)
    new_x = np.array(new_x)
    # Standard scale everything
    new_x = (new_x-np.mean(new_x, axis=0))/np.std(new_x, axis=0)
    return new_x,y


def getXY_evalsplit():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for fnum in range(1,15):
        if fnum==6:
            continue
        if fnum >= 10:
            fstr = fnum
        else:
            fstr = f"0{fnum}"
        print(fstr)
        x,y = loadXY(f"flm_XY/flm_resting_state_block{fstr}.csv") 

        X_train.append(x[len(x)//4:])
        X_test.append(x[:len(x)//4])
        y_train.append(y[len(x)//4:])
        y_test.append(y[:len(x)//4])
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    return X_train, y_train, X_test, y_test

x_t, y_t, x_e, y_e = getXY_evalsplit()
x_t, y_t = preprocessXY(x_t, y_t)
x_e, y_e = preprocessXY(x_e, y_e)
#clf = sklearn.en
print(f"X_train: {np.shape(x_t)}, y_train:{np.shape(y_t)}")
print(f"X_test: {np.shape(x_e)}, y_test:{np.shape(y_e)}")
print(f"Mean of x_train: {np.sum(np.mean(x_t, axis=0))}, std of x_train: {np.mean(np.std(x_t, axis=0))}")
print(np.sum(y_t)/len(y_t), np.sum(y_e)/len(y_e))
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100, n_jobs=16)
clf = clf.fit(x_t, y_t)
y_pred_t = clf.predict(x_t)
y_pred = clf.predict(x_e)
print('The Accuracy on the train set is %.2f%%' %(sum(y_t==y_pred_t)/len(y_t)*100))
print('The Accuracy on the test set is %.2f%%' %(sum(y_e==y_pred)/len(y_e)*100))
