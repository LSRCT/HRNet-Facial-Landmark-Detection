
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

    def load_pic_path(self, pic_path):
        pic_og = Image.open(pic_path)
        pic = pic_og.resize((256,256))
        pic = np.asarray(pic).copy()
        return pic


    def inf_pic(self, pic):
        """
        Infer the facial landmarks for a single picture. Should already be cut
        """
        pic_tens = self.preprocess_pic(pic)
        with torch.no_grad():
            pic_out = self.model(pic_tens)
            score_map = pic_out.data.cpu()
            preds = decode_preds(score_map, [np.array([128,128] )], [1.28], [64, 64])[0]
        return preds.numpy()
        #plt.imshow(pic)
        #plt.scatter(preds[:,0], preds[:,1], 1)
        #plt.show()
        

    def preprocess_pic(self, pic):
        """
        Convert the a picture to the format needed for the model
        TODO do this with some model
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



if __name__ == '__main__':
    cfg_file = "experiments/wflw/face_alignment_wflw_hrnet_w18.yaml"
    model_file = "HR18-WFLW.pth"
    pic_path = "r_s.png"
    flm = BSD_FLM(cfg_file, model_file)
    flm.inf_pic(pic_path)

