"""Runner Module"""

import time

import cv2
import torch

from constants import PROGRESS_NUM
from helpers.utility import Utility
from modules.bicubic import BicubicInterpolation
from modules.real_esrgan import RealESRGAN

class Runner: 

    def __init__(self): 
        self.utility = Utility()

    def run_bicubic_interpolation(self, targets): 

        bicubic_interpolation = BicubicInterpolation()

        for target in targets:
            print(f'\t{target["name"]}')

            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['test_dir'])

            # Filter the images that have already been processed
            result_imgs = self.utility.get_files_in_dir(target['results_dir'])
            target_imgs = [img for img in target_imgs if img not in result_imgs]

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

                    if (idx) % 10 == 0:
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

                    if checkpoint !=0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx+=1

                print(f'\t\tSuccessfully upscaled images. Can be found in {target["results_dir"]}')
