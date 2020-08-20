import sklearn
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import numpy as np
import sys, os
import ast
import cv2


def loadX(path):
    X = []
    with open(path, "r") as f:
        # first one is header
        f.readline()
        # other ones are data
        for line in f:
            ls = line.strip("\n")
            ls = ls.split(";")[1:]
            ls = [[float(x) for x in coord.split(",")] for coord in ls[:-1]]
            X.append(ls)
    return np.array(X)

def loadY(path):
    Y = []
    with open(path, "r") as f:
        # first one is header
        f.readline()
        # other ones are data
        for line in f:
            ls = line.strip("\n")
            ls = ls.split(";")[1:]
            label= int(ls[-1])
            Y.append(label)
    return np.array(Y)

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
    x_mean = np.mean(new_x, axis=0)
    x_std = np.std(new_x, axis=0)
    new_x = (new_x-x_mean)/x_std
    return new_x,y


def getXY_exp1(XY_split):
    """
    Get a train/test split of the data.
    :param XY_split: String specifying the kind of data split
                     "full": all data
                     "no10": Exclude block 10
                     "eyesopen": Only blocks with eyes open
                     "eyesclosed": Only block with eyes closed
                     "eyesclosedno10": Only block with eyes closed, exclude block 10
    """

    if XY_split == "full":
        frange = list(range(1,15))
    elif XY_split == "no10":
        frange = [1,2,3,4,5,6,7,8,9,11,12,13,14]
    elif XY_split == "eyesopen":
        frange = [1, 3, 5, 7, 9, 11, 13]
    elif XY_split == "eyesclosed":
        frange = [2, 4, 6, 8, 10, 12, 14]
    elif XY_split == "eyesclosedno10":
        frange = [2, 4, 6, 8, 12, 14]
    return getXY_exp1_range(frange)

def getXY_exp1_range(block_range):
    """
    Get a train/test split on the data using only specified blocks
    1st and 3rd quarter for earch block is train data,
    2nd and 4th quarter for each block is test data
    :param block_range: List containing the blocks to be considered
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for fnum in block_range:
        if fnum >= 10:
            fstr = fnum
        else:
            fstr = f"0{fnum}"
        x = loadX(f"flmXY_2/flm_resting_state_block{fstr}.csv") # TODO please stop hardcoding paths
        y = loadY(f"flm_labels/labels_resting_state_block{fstr}.csv") 
        print(f"Block {fstr} X: {np.shape(x)}, Y: {np.shape(y)}")
        
        X_train.append(x[:len(x)//4])
        y_train.append(y[:len(x)//4])
        X_train.append(x[len(x)//2:(3*len(x))//4])
        y_train.append(y[len(x)//2:(3*len(x))//4])
        
        X_test.append(x[len(x)//4:len(x)//2])
        y_test.append(y[len(y)//4:len(y)//2])
        X_test.append(x[(3*len(x))//4:])
        y_test.append(y[(3*len(y))//4:])
            
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    return X_train, y_train, X_test, y_test

def getXY_exp3(XY_split):
    """
    Get a train/test split of the data.
    :param XY_split: String specifying the kind of data split
                     "full": all data
                     "no10": Exclude block 10
                     "eyesopen": Only blocks with eyes open
                     "eyesclosed": Only block with eyes closed
    """

    if XY_split == "full":
        test_blocks = [5, 6, 9, 10, 13, 14]
        train_blocks = [3,4,7,8, 11, 12]
    elif XY_split == "no10":
        test_blocks = [5, 6, 9, 13, 14]
        train_blocks = [3,4,7,8,11, 12]
    elif XY_split == "eyesopen":
        #test_blocks= [5, 9, 13]
        train_blocks= [5, 9, 13]
        test_blocks = [3, 7, 11]
        #train_blocks = [3, 7, 11]
    elif XY_split == "eyesclosed":
        test_blocks = [8, 14]
        train_blocks = [4, 6, 12]
    elif XY_split == "probt":
        test_blocks = [13]
        #train_blocks = [3,5, 7, 9,11]
        train_blocks = [3, 7, 11]
    return test_blocks, train_blocks

def getXY_exp3_range(test_b, train_b):
    """
    Get a test/train split of the data, specifying the block for both categories
    :param test_b: List of blocks for the test set
    :param train_b: List of blocks for the train set
    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    b_lengths_t= []
    b_lengths_e= []
    for fnum in test_b+train_b:
        if fnum >= 10:
            fstr = fnum
        else:
            fstr = f"0{fnum}"
        x = loadX(f"flmXY_2/flm_resting_state_block{fstr}.csv") # TODO please stop hardcoding paths
        y = loadY(f"flm_labels/labels_resting_state_block{fstr}.csv") 
        print(f"Block {fstr} X: {np.shape(x)}, Y: {np.shape(y)}")
        
       #x_1 = x[:len(x)//4]
       #y_1 = y[:len(y)//4]
       #x_2 = x[(2*len(x))//4:(3*len(x))//4]
       #y_2 = y[(2*len(y))//4:(3*len(x))//4]

       #x_1 = x[len(x)//4:len(x)//2]
       #y_1 = y[len(y)//4:len(y)//2]
       #x_2 = x[(3*len(x))//4:]
       #y_2 = y[(3*len(y))//4:]
       #x = np.concatenate([x_1, x_2])
       #y = np.concatenate([y_1, y_2])
        
        if fnum in test_b:
            X_test.append(x)
            y_test.append(y)
            b_lengths_e.append(len(y))
        else:  
            X_train.append(x)
            y_train.append(y)
            b_lengths_t.append(len(y))

    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    return X_train, y_train, X_test, y_test, b_lengths_t, b_lengths_e

def ica_data(xt, xe):
    from sklearn.decomposition import FastICA
    ica = FastICA(max_iter=1000, whiten=False)
    xt = ica.fit_transform(xt)
    xe = ica.transform(xe)
    xt = (xt-np.mean(xt, axis=0))/np.std(xt, axis=0)
    xe = (xe-np.mean(xe, axis=0))/np.std(xe, axis=0)
    #ica.fit(xt)
    #xt = np.dot(xt, pca.components_[1:2].T)
   #plt.plot(pca.explained_variance_ratio_)
   #plt.title("PCA explained variance")
   #plt.xlabel("Component")
   #plt.ylabel("Explained variance in %")
   #plt.show()
   #f_weights = pca.components_[2].reshape((-1,2))
   #f_weights = np.mean(f_weights, axis=1)
   #plt.plot(f_weights)
   #plt.show()
    return xt, xe

def pca_data(xt, xe):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=30)
    xt = pca.fit_transform(xt)
    xe = pca.transform(xe)
    #xt = np.dot(xt, pca.components_[2:20].T)
    plt.plot(pca.explained_variance_ratio_)
    plt.title("PCA explained variance")
    plt.xlabel("Component")
    plt.ylabel("Explained variance in %")
    plt.show()
    f_weights = pca.components_[0].reshape((-1,2))
    f_weights = np.mean(f_weights, axis=1)
    plt.title("Principal component 1")
    plt.xlabel("Weight")
    plt.ylabel("Feature")
    plt.plot(f_weights)
    plt.show()
    return xt, xe

    
def save_clf(y_true, y_pred, csv_name):
    y_pred = [f"{round(x[1],2)}" for x in y_pred]
    save_name = csv_name
    print(f"Saving to csv {save_name}")
    with open(save_name, "w") as fcsv:
        header = "frame;"+"label;"+"pred"+"\n"
        fcsv.write(header)
        for frame_numb, (Y, Y_p) in enumerate(zip(y_true, y_pred)):
            row = str(frame_numb)+";"+ str(Y)+";" +str(Y_p)+ "\n"   
            fcsv.write(row)
    print(f"Done")
    
    

### Prepare the data ###

# full, no10, eyesopen, eyesclosed
e_blocks, t_blocks = getXY_exp3("eyesopen")
x_t, y_t, x_e, y_e, len_t, len_e = getXY_exp3_range(e_blocks, t_blocks)

x_t, y_t = preprocessXY(x_t, y_t)

x_e, y_e = preprocessXY(x_e, y_e)

x_t, x_e = pca_data(x_t, x_e)
#x_t, x_e = ica_data(x_t, x_e)
print(np.shape(x_t))
#x_e = ica_data(x_e)
print(f"X_train: {np.shape(x_t)}, y_train:{np.shape(y_t)}")
print(f"X_test: {np.shape(x_e)}, y_test:{np.shape(y_e)}")
print(f"Mean of x_train: {np.sum(np.mean(x_t, axis=0))}, std of x_train: {np.mean(np.std(x_t, axis=0))}")
print(np.sum(y_t)/len(y_t), np.sum(y_e)/len(y_e))


### Classify ###

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
clf = RandomForestClassifier(n_estimators=1000, n_jobs=16, min_impurity_decrease=0.005)
clf = clf.fit(x_t, y_t)
y_pred_t = clf.predict(x_t)
y_pred = clf.predict(x_e)
print('The Accuracy on the train set is %.2f%%' %(sum(y_t==y_pred_t)/len(y_t)*100))
print('The Accuracy on the test set is %.2f%%' %(sum(y_e==y_pred)/len(y_e)*100))
prob= clf.predict_proba(x_e)
prob_t = clf.predict_proba(x_t)

exp_path = "exp3"

b_pos = 0
for blk_num, (blk, b_len) in enumerate(zip(t_blocks, len_t)):
    if blk < 10:
        blk_str = f"0{blk}" 
    else:
        blk_str = f"{blk}" 
    csv_p = f"{exp_path}/pred/pred_flm_block{blk_str}.csv"
    blk_range = range(b_pos, b_pos+b_len)
    b_pos += b_len
    save_clf(y_t[blk_range], prob_t[blk_range], csv_p)

b_pos = 0
for blk_num, (blk, b_len) in enumerate(zip(e_blocks, len_e)):
    if blk < 10:
        blk_str = f"0{blk}" 
    else:
        blk_str = f"{blk}" 
    csv_p = f"{exp_path}/pred/pred_flm_block{blk_str}.csv"
    blk_range = range(b_pos, b_pos+b_len)
    b_pos += b_len
    save_clf(y_e[blk_range], prob[blk_range], csv_p)


### Bootstrap ### 

np.random.shuffle(y_e)
np.random.shuffle(y_t)
print('The BS on the train set is %.2f%%' %(sum(y_t==y_pred_t)/len(y_t)*100))
print('The BS on the test set is %.2f%%' %(sum(y_e==y_pred)/len(y_e)*100))

import sys
sys.exit()

from sklearn.inspection import permutation_importance
result = permutation_importance(clf, x_e, y_e, n_jobs=16, n_repeats=10)

f_imp = result.importances_mean
f_imp = np.reshape(f_imp, (-1, 2))
f_imp = np.average(f_imp, axis=-1)
f_names = ["round", "brows", "nose", "eyes", "mouth"]
f_mag = []
f_mag.append(np.sum(f_imp[:33])) # face shape
f_mag.append(np.sum(f_imp[33:51])) # brows
f_mag.append(np.sum(f_imp[51:60])) # nose
f_mag.append(np.sum(f_imp[60:76])) # eyes
f_mag.append(np.sum(f_imp[76:96])) # mouth

f_mag[3] += np.sum(f_imp[96:98]) # eye centers
plt.bar(f_names, f_mag) 
plt.title("Feature importances")
plt.show()

