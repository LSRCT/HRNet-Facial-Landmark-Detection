
import numpy as np
import os
import argparse

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
        Cut the pic to the facial boundary box. TODO dont hardcode this
        :param pic: picture to cut. should be and ndarray
        :return: cut pic as ndarray
        """
        pic = pic[80:330, 250:478]
        return pic


    def proc_video(self, path, show=0, save_avi=0, save_csv=0):
        """
        Infer landmarks for a video
        :param path: Location of the video on hard disk
        :param show: Display the video with landmarks live.
        :param save_avi: Save the landmarks added to the original video as avi.
        :param save_csv: Save the landmarks to a csv file.j
        """
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
            frame = self.cut_fbb(frame)
            # resize to correct shape
            pic = cv2.resize(frame, (256,256))
            # get facial landmarks
            preds = flm.inf_pic(pic)
            
            pred_list.append(preds)
            print(f"Done with frame {len(pred_list)}")
            # draw landmarks as circles
            if show or save_avi:
                for pred in preds:
                    cv2.circle(pic, (pred[0], pred[1]), 1, (0,0,255), -1) 
            if save_avi:
                out.write(pic)
            if show:
                # show image, press q to stop the video display
                cv2.imshow('Frame', pic) 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            ret, frame = cap.read()
        cap.release()
        
        if save_csv:
            save_name = "flm_"+path.split("/")[-1][:-4]+".csv"
            print(f"Saving to csv {save_name}")
            with open(save_name, "w") as fcsv:
                header = "frame;"+"".join(["lm"+str(x)+";" for x in range(len(pred_list[0]))])+"\n"
                fcsv.write(header)
                for frame_numb, lm in enumerate(pred_list):
                    row = str(frame_numb)+";" + "".join([str(coord)+";" for coord  in lm]) + "\n"   
                    fcsv.write(row)
        if show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    cfg_file = "experiments/wflw/face_alignment_wflw_hrnet_w18.yaml"
    model_file = "HR18-WFLW.pth"
    avi_path = "/media/lsrct/Data/BSDLAB/FacialLMs/resting_state/resting_state_block01.avi"
    pic_path = "r_s.png"
    flm = BSD_FLM(cfg_file, model_file)
    flm.proc_video(avi_path, save_csv=1)
    #flm.inf_pic(pic_path)

