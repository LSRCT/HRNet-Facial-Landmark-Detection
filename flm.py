
from collections import OrderedDict
import numpy as np
import os
import argparse

from facenet_pytorch import MTCNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
from lib.core.evaluation import decode_preds
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class BSD_FLM:
    def __init__(self,cfg_path, model_path):
        parser = argparse.ArgumentParser(description='Train Face Alignment')

        parser.add_argument('--cfg', help='experiment configuration filename',
                            required=True, type=str)
        parser.add_argument('--model-file', help='model parameters', required=True, type=str)

        args = parser.parse_args(["--cfg",cfg_path, "--model-file", model_path])
        update_config(config, args)

        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()
        model = models.get_face_alignment_net(config)

        # this is important to run on CPU
        model = nn.DataParallel(model)
        model = model.cpu()

        # load model
        state_dict = torch.load(args.model_file, map_location="cpu")
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            model.module.load_state_dict(state_dict)
        model.eval()
        self.model = model
        
        # init the stuff for facial boundary detection
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))
        self.fbb_model = MTCNN(device=device, select_largest=False, thresholds=[0.5, 0.6, 0.6]) 
        #self.fbb_init = [315, 95, 500, 315]
        self.fbb_init = []

    def inf_pic(self, pic):
        """
        Infer the facial landmarks for a single picture. Should already be cut
        :param pic: picture to get landmarks for. should be an ndarray
        :return: Numpy array of landmark coordinates
        """
        pic_tens = self.preprocess_pic(pic)
        with torch.no_grad():
            pic_out = self.model(pic_tens)
            score_map = pic_out.data.cpu()
            preds = decode_preds(score_map, [np.array([128,128] )], [1.28], [64, 64])[0]
        return preds.numpy()
        

    def preprocess_pic(self, pic):
        """
        Convert the a picture to the format needed for the model
        :param pic: picture to preprocess. should be and ndarray
        :return: preprocessed pic as ndarray
        """
        # mean and std of the training dataset
        pic_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        pic_std = np.array([0.229, 0.224, 0.225], dtype=np.float32) 
        # std scale it
        pic_tp = (pic/255.0-pic_mean)/pic_std    
        # make it a pytorch tensor
        pic_tp = pic_tp.transpose([2,0,1])
        pic_tens = torch.from_numpy(pic_tp).type("torch.FloatTensor")
        pic_tens = pic_tens.unsqueeze(0)
        return pic_tens

    def cut_fbb(self,pic):
        """
        Cut the pic to the facial boundary box.
        :param pic: picture to cut. should be and ndarray
        :return: cut pic as ndarray
        """
        # only update FBB if there is no existing FBBB
        if len(self.fbb_init) == 0:
            # Calculate facial boundary box
            fbb_pred = self.fbb_model.detect(pic)
            self.fbb_init = [int(x) for x in fbb_pred[0][0]]
        #pic = pic[80:330, 300:508]
        fbb = self.fbb_init
        # margin so the person can move a bit
        margin = 50
        pic = pic[fbb[1]-margin:fbb[3]+margin, fbb[0]-margin:fbb[2]+margin]
        return pic
        

    def get_labels(self, path, length):
        print(f"Calculating labels for {path}")
        trig_file = path[:-4]+".trigger"
        save_name = "flm_"+path.split("/")[-1][:-4]+".csv"
        tf_cont = OrderedDict()
        with open(trig_file) as tf:
            for line in tf.readlines():
                ls = line.strip("\n").split(",")
                tf_cont[int(ls[1])] = float(ls[0])
        print(tf_cont)
        if 239 in tf_cont:
            start =int((tf_cont[239]-tf_cont[10])*20)
            stop = int((tf_cont[199]- tf_cont[10])*20)
        else:
            start = length
            stop = length
        labels = [0]*start+[1]*(stop-start)+[0]*(length-stop)
        print(f"{start} frames off, {stop-start} frames on, {length-stop} frames off")
        print(f"Done")
        return labels


    def save_XY_pair(self, x,y, og_filename):
        """
        Convenience function to save a feature list plus a label list to csv
        """
        save_name = "flm_"+og_filename.split("/")[-1][:-4]+".csv" # TODO path
        print(f"Saving to csv {save_name}")
        with open(save_name, "w") as fcsv:
            header = "frame;"+"".join(["lm"+str(x)+";" for x in range(len(x[0]))])+"label"+"\n"
            fcsv.write(header)
            for frame_numb, lm in enumerate(x):
                row = str(frame_numb)+";" + "".join([str(coord[0])+","+str(coord[1])+";" for coord  in lm]) +str(y[frame_numb]) + "\n"   
                fcsv.write(row)
        print(f"Done")
        

    def save_labels(self, labels, og_filename):
        """
        Save labels to a csv file
        """
        save_name = "labels_"+og_filename.split("/")[-1][:-4]+".csv" 
        print(f"Saving to csv {save_name}")
        with open(save_name, "w") as fcsv:
            header = "frame;"+"label"+"\n"
            fcsv.write(header)
            for frame_numb, Y in enumerate(labels):
                row = str(frame_numb)+";"+ str(Y) + "\n"   
                fcsv.write(row)
        print(f"Done")

        

    def get_video_length(self, path):
        """
        Get the number of frames in a given video file.
        Note that this is inefficient as TODO
        :param path: Path to the video file
        :return: Number of frames
        """
        frames = 0

        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        while ret:
            frames += 1
            ret, frame = cap.read()
        cap.release()
        return frames

    def proc_video(self, path, show=0, save_avi=0, save_csv=0):
        """
        Infer landmarks for a video
        :param path: Location of the video on hard disk
        :param show: Display the video with landmarks live.
        :param save_avi: Save the landmarks added to the original video as avi.
        :param save_csv: Save the landmarks to a csv file.
        """
        print(f"Processing {path}")
        # create a ne boundary box every video
        self.fbb_init = []
        
        # cv2 and Pillow use different color sutff
        cv2.CAP_PROP_CONVERT_RGB = False
        cap = cv2.VideoCapture(path)
        if save_avi:
            save_name = "flm_"+path.split("/")[-1]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_name, fourcc, 20.0,(256,256))

        ret, frame = cap.read()
        pred_list = []
        # for every frame
        while ret:
            # crop it so only the face is tehre
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = self.cut_fbb(frame)
            # resize to correct shape
            pic = cv2.resize(frame, (256,256))
            # get facial landmarks
            preds = flm.inf_pic(pic)
            
            pred_list.append(preds)
            if len(pred_list)%100 == 0:
                print(f"Done with {len(pred_list)} Frames")
            # draw landmarks as circles
            if show or save_avi:
                for pred in preds:
                    cv2.circle(pic, (pred[0], pred[1]), 1, (0,0,255), -1) 
            pic = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
            if save_avi:
                out.write(pic)
            if show:
                # show image, press q to stop the video display
                cv2.imshow('Frame', pic) 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            ret, frame = cap.read()
        cap.release()        
        if show:
            cv2.destroyAllWindows()
        print(f"Done")
        return pred_list


if __name__ == '__main__':
    cfg_file = "experiments/wflw/face_alignment_wflw_hrnet_w18.yaml"
    model_file = "HR18-WFLW.pth"
    flm = BSD_FLM(cfg_file, model_file)
    for fnum in range(1,15):
        if fnum >= 10:
            fstr = fnum
        else:
            fstr = f"0{fnum}"
        avi_path = f"/mnt/d/BSDLAB/FacialLMs/resting_state/resting_state_block{fstr}.avi"
        print(avi_path)
        #features = flm.proc_video(avi_path, save_avi=0, show=0)
        n_frames = flm.get_video_length(avi_path)
        labels = flm.get_labels(avi_path, n_frames)
        flm.save_labels(labels, avi_path)
        #flm.save_XY_pair(features,labels, avi_path)

