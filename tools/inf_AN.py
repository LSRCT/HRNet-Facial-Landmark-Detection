# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------
import numpy as np
import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
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

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model)
    model = model.cpu()

    # load model
    state_dict = torch.load(args.model_file, map_location="cpu")
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    pic_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    pic_std = np.array([0.229, 0.224, 0.225], dtype=np.float32) 
    print(np.shape(test_loader.dataset[0][0]))
    pic = test_loader.dataset[0][0]
    pic_og = Image.open("an2.jpg")
    pic = pic_og.resize((256,256))
    pic = np.asarray(pic).copy()
    pic_tp = (pic/255.0-pic_mean)/pic_std    
    pic_tp = pic_tp.transpose([2,0,1])
    pic_tens = torch.from_numpy(pic_tp).type("torch.FloatTensor")
    pic_tens = pic_tens.unsqueeze(0)
    

    print(np.shape(pic_tens))
    model.eval()
    with torch.no_grad():
        pic_out = model(pic_tens)
        score_map = pic_out.data.cpu()
        preds = decode_preds(score_map, [np.array([128,128] )], [1.28], [64, 64])[0]
        print(np.shape(preds))
    print(preds)

    plt.imshow(pic)
    plt.scatter(preds[:,0], preds[:,1], 1)
    plt.show()
    #nme, predictions = function.inference(config, test_loader, model)

    #torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()

