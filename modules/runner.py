"""Runner Module"""

import time

import cv2
import torch
import numpy as np

from constants import PROGRESS_NUM
from helpers.utility import Utility
from modules.bicubic import BicubicInterpolation
from modules.real_esrgan import RealESRGAN
from modules.SRCNN import SRCNN

class Runner: 

    def __init__(self): 
        self.utility = Utility()

    def run_bicubic_interpolation(self, targets):
        """Run bicubic interpolation in the image files provided

        Args:
            targets (list): list of directories with images
        """

        bicubic_interpolation = BicubicInterpolation()

        # Loop over all the image directories
        for target in targets:
            print(f'\t{target["name"]}')

            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['test_dir'])

            # Filter the images that have already been processed
            result_imgs = self.utility.get_files_in_dir(target['results_dir'])
            target_imgs = [img for img in target_imgs if img not in result_imgs]

            checkpoint = int(len(target_imgs)/PROGRESS_NUM)

            # If all the images have been processed, then skip
            if len(target_imgs) == 0:
                print(f'\t\tAlready upscaled images. Can be found in {target["results_dir"]}')
            else:
                print(f'\t\tUpscaling {len(target_imgs)} images...')

                for img_name in target_imgs:

                    bicubic_interpolation.run(
                        input=f'{target["test_dir"]}/{img_name}',
                        output=f'{target["results_dir"]}/{img_name}',
                        scale=target['scale']
                    )

                    if checkpoint!= 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1

                print(f'\t\tSuccessfully upscaled images. Can be found in {target["results_dir"]}')

    def run_real_esrgan(self, targets): 
        for target in targets:
            print(f'\t{target["name"]}')

            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['test_dir'])

            # Filter the images that have already been processed
            result_imgs = self.utility.get_files_in_dir(target['results_dir'])
            target_imgs = [img for img in target_imgs if img not in result_imgs]

            checkpoint = int(len(target_imgs)/PROGRESS_NUM)

            # If all the images have been processed, then skip
            if len(target_imgs) == 0:
                print(f'\t\tAlready upscaled images. Can be found in {target["results_dir"]}')
            else:
                print(f'\t\tUpscaling {len(target_imgs)} images...')

                for img_name in target_imgs:
                    real_esrgan = RealESRGAN(device='cuda')
                    real_esrgan.run(
                        input = f'{target["test_dir"]}/{img_name}',
                        output = f'{target["results_dir"]}/{img_name}'
                    )
                    torch.cuda.empty_cache()

                    # Print the progress
                    if checkpoint !=0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx+=1

                print(f'\t\tSuccessfully upscaled images. Can be found in {target["results_dir"]}')

    def run_srcnn(self, targets):
        for target in targets:
            print(f'\t{target["name"]}')

            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['test_dir'])[:1]

            # Filter the images that have already been processed
            result_imgs = self.utility.get_files_in_dir(target['results_dir'])
            target_imgs = [img for img in target_imgs if img not in result_imgs]

            checkpoint = int(len(target_imgs)/PROGRESS_NUM)

            # If all the images have been processed, then skip
            if len(target_imgs) == 0:
                print(f'\t\tAlready upscaled images. Can be found in {target["results_dir"]}')
            else:
                print(f'\t\tProcessing {len(target_imgs)} images...')

                srcnn = SRCNN()

                for img_name in target_imgs:
                    img = cv2.imread(f'{target["test_dir"]}/{img_name}')

                    print(img.shape)
                    img = self.utility.mod_crop(img, target['scale'])
                    
                    y_cr_cb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

                    test_img = np.zeros((1, y_cr_cb_img.shape[0], y_cr_cb_img.shape[1], 1), dtype=float)
                    test_img[0, :, :, 0] = y_cr_cb_img[:, :, 0].astype(float) / 255
                    
                    pred_img = srcnn.predict(test_img)

                    pred_img *= 255 
                    pred_img[pred_img[:] > 255] = 255 
                    pred_img[pred_img[:] < 0 ] = 0
                    pred_img = pred_img.astype(np.uint8)

                    y_cr_cb_img = self.utility.shave(y_cr_cb_img, target['border'])
                    y_cr_cb_img[:, :, 0] = pred_img[0, :, :, 0]

                    pred_img = cv2.cvtColor(y_cr_cb_img, cv2.COLOR_YCrCb2BGR)

                    cv2.imwrite(f'{target["results_dir"]}/{img_name}', pred_img)

                    # Print the progress
                    if checkpoint != 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1
                
                print(f'\t\tSuccessfully upscaled images. Can be found in {target["results_dir"]}')


