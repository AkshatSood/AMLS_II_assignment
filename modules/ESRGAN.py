"""ESRGAN Module"""

import glob
import os.path as osp

import cv2
import numpy as np
import torch

from constants import MODEL_RRDB_ESRGAN_X4
from modules.RRDBNet_arch import RRDBNet


class ESRGAN():

    def __init__(self, device='cuda', model_path=MODEL_RRDB_ESRGAN_X4):
        """Constructor

        References code provided at https://github.com/xinntao/ESRGAN

        Args:
            device (str, optional): Which hardware to use for pytorch. Defaults to 'cuda'.
            model_path (str, optional): Path the to the pretrained model weights. Defaults to MODEL_RRDB_ESRGAN_X4.
        """
        self.device = torch.device(device)
        self.model = RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    def run(self, input, output):
        """Upsacles the provided image with the ESRGAN (RRDB) model, 
        and saves the image in the provided location

        Args:
            input (str): Input image path
            output (str): Output image path
        """

        # Read and normalise the image
        img = cv2.imread(input, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        # Upscale the image using the RRDB model
        with torch.no_grad():
            upscaled_img = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        upscaled_img = np.transpose(upscaled_img[[2, 1, 0], :, :], (1, 2, 0))
        upscaled_img = (upscaled_img * 255.0).round()
        
        # Save the upscaled image
        cv2.imwrite(output, upscaled_img)
