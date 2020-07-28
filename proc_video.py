import time
import cv2
from flm import BSD_FLM
import numpy as np


if __name__ == '__main__':
    cfg_file = "experiments/wflw/face_alignment_wflw_hrnet_w18.yaml"
    model_file = "HR18-WFLW.pth"
    pic_path = "r_s.png"
    flm = BSD_FLM(cfg_file, model_file)
    cap = cv2.VideoCapture('/media/lsrct/Data/BSDLAB/FacialLMs/resting_state/resting_state_block01.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("flm_out.avi", fourcc, 20.0,(256,256))
    ret, frame = cap.read()
    #while cap.isOpened():
    while ret:
        frame = frame[80:330, 250:478]
        pic = cv2.resize(frame, (256,256))
        #pic = np.asarray(Ipic)
        preds = flm.inf_pic(pic)
        for pred in preds:
            cv2.circle(pic, (pred[0], pred[1]), 1, (0,0,255), -1) 
        out.write(pic)
        #cv2.imshow('Frame', pic) 
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    flm.inf_pic(pic_path)
