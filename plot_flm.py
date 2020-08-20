import sklearn
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import numpy as np
import sys, os
import ast
import cv2


def loadPred(path):
    Y = []
    pred = []
    with open(path, "r") as f:
        # first one is header
        f.readline()
        # other ones are data
        for line in f:
            ls = line.strip("\n")
            ls = ls.split(";")[1:]
            label= int(ls[-2])
            p = float(ls[-1])
            Y.append(label)
            pred.append(p)
    return np.array(Y), np.array(pred)



def annotate_video(p_true, p_pred, vid_path, s_path):
    p_p = [f"Pred: {x}" for x in p_pred]
    p_t = [f"True: {x}" for x in p_true]

    cv2.CAP_PROP_CONVERT_RGB = False
    cap = cv2.VideoCapture(vid_path)
    save_name = f"{s_path}//" + "an_"+vid_path.split("/")[-1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_name, fourcc, 20.0,(256,256))

    ret, frame = cap.read()
    pred_list = []
    fnum = 0
    # for every frame
    while ret:
        pic = frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pic, p_p[fnum],(150,20), font, 0.5,(255,255,255), 1,cv2.LINE_AA)
        cv2.putText(pic, p_t[fnum],(150,50), font, 0.5,(255,255,255), 1,cv2.LINE_AA)
        out.write(pic)
        fnum += 1
        ret, frame = cap.read()
    cap.release()        


block_str = "07"
exp_folder = "exp3"
csv_p = f"{exp_folder}/pred/pred_flm_block{block_str}.csv"
y_e, prob = loadPred(csv_p)
avi_path = f"flmXY_2/flm_resting_state_block{block_str}.avi"
annotate_video(y_e, prob, avi_path, exp_folder)


### Plot probabilities ###

plt.plot(y_e, label="Ground truth")
plt.title(f"Predicted probabilities block {block_str}")
ravg = 50
prob = np.convolve(prob, np.ones(ravg)) / ravg
plt.plot(prob, label="Predicted probability")
plt.ylabel("P(y=1)")
plt.xlabel("frame #")
plt.legend(loc=1)
plt.savefig(f"{exp_folder}/{block_str}_pred.png")

