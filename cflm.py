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
            #ls = ls.replace(".", "")
            #ls = ls.replace("[", "")
            #ls = ls.replace("]", "")
            ls = ls.split(";")[1:]
            label= int(ls[-1])
            ls = [[float(x) for x in coord.split(",")] for coord in ls[:-1]]
            #ls = [[float(coord[:3].strip(" ")), float(coord[3:].strip(" "))] for coord in ls[:-1]]
            X.append(ls)
            Y.append(label)
    return np.array(X), np.array(Y)


def preprocessXY(x,y):
    # Lowpass filter to get rid of wiggling
    #x = uniform_filter1d(x, 10, axis=0)
    new_x = [] 
    for frame in x:
        f_new = frame
        
        # center the face basically
        f_new = f_new - np.mean(f_new, axis=0)
        
        # account for being closer to camera
        f_new = f_new.T
        f_new[0] = f_new[0]/np.max(f_new[0])
        f_new[1] = f_new[1]/np.max(f_new[1])
        f_new = f_new.T

        #f_new = np.linalg.norm(frame, axis=1)
        f_new = f_new.flatten()
        new_x.append(f_new)
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
        if fnum >= 10:
            fstr = fnum
        else:
            fstr = f"0{fnum}"
        # for testing eyes open/closed
        if fnum in [2,4,6, 8, 10, 12, 14]:
        #if fnum in [1,3,5, 7, 9, 11, 13]:
            continue
        # for testing without block 10
        #if fnum == 10:
        #    continue
        x,y = loadXY(f"flmXY_2/flm_resting_state_block{fstr}.csv") 
        if fnum == 1:
            continue
        if fnum in [5, 9, 13]:
        #if fnum in [3, 7, 11]:
            X_test.append(x)
            y_test.append(y)
        else:  
            X_train.append(x)
            y_train.append(y)
       #X_train.append(x[len(x)//4:])
       #X_test.append(x[:len(x)//4])
       #y_test.append(y[:len(x)//4])
       #y_train.append(y[len(x)//4:])
        print(fstr)
        print(np.shape(np.mean(x, axis=0)))
            
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    return X_train, y_train, X_test, y_test

x_t, y_t, x_e, y_e = getXY_evalsplit()
x_t, y_t = preprocessXY(x_t, y_t)

x_e, y_e = preprocessXY(x_e, y_e)

print(f"X_train: {np.shape(x_t)}, y_train:{np.shape(y_t)}")
print(f"X_test: {np.shape(x_e)}, y_test:{np.shape(y_e)}")
print(f"Mean of x_train: {np.sum(np.mean(x_t, axis=0))}, std of x_train: {np.mean(np.std(x_t, axis=0))}")
print(np.sum(y_t)/len(y_t), np.sum(y_e)/len(y_e))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
clf = RandomForestClassifier(n_estimators = 1000, n_jobs=16)
clf = clf.fit(x_t, y_t)
y_pred_t = clf.predict(x_t)
y_pred = clf.predict(x_e)
print('The Accuracy on the train set is %.2f%%' %(sum(y_t==y_pred_t)/len(y_t)*100))
print('The Accuracy on the test set is %.2f%%' %(sum(y_e==y_pred)/len(y_e)*100))
c_m = confusion_matrix(y_e, y_pred)
print(c_m)

# bootstrap
np.random.shuffle(y_e)
np.random.shuffle(y_t)
print('The BS on the train set is %.2f%%' %(sum(y_t==y_pred_t)/len(y_t)*100))
print('The BS on the test set is %.2f%%' %(sum(y_e==y_pred)/len(y_e)*100))
