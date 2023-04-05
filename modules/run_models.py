"""Run Models Module"""

import time

import torch
import torch.backends.cudnn as cudnn

from constants2 import (MODEL_RRDB_ESRGAN_X4, MODEL_RRDB_PSNR_X4, PROGRESS_NUM,
                        RDDBESRGAN, RDDBPSNR, TARGETS_RRDB)
from helpers.utility import Utility
from modules.real_esrgan import RealESRGAN
from modules.runner import Runner


class RunModels:

    def __init__(self):
        self.utility = Utility()
        self.runner = Runner()

    def run_rrdb_esrgan(self):
        print('\n=> Running RRDB (ESRGAN) on test datasets...')

        for step, target in enumerate(TARGETS_RRDB):
            print(f'\n\t[{step+1}/{len(TARGETS_RRDB)}] Running RRDB (ESRGAN) (X4) on {target["dataset"]} dataset...')

            RES_DIR = target['res_dir'] + '/RRDB_ESRGAN'

            print(f'\t\tSource directory: {target["src_dir"]}')
            print(f'\t\tResults directory: {RES_DIR}')

            self.utility.check_and_create_dir(RES_DIR)

            target_imgs = self.utility.get_imgs_with_tag_from_dir(dir_path=target['src_dir'], tag=target['src_tag'])[:10]

            result_imgs = self.utility.get_files_in_dir(RES_DIR)

            target_imgs = self.utility.filter_names_ignore_tag(src=target_imgs, res=result_imgs)

            if len(target_imgs) == 0:
                print('\t\tAlready upscaled images. Skipping this step.')
            else:
                start_time = time.time()
                idx = 1
                checkpoint = int(len(target_imgs)/PROGRESS_NUM)

                print(f'\t\tUpscaling {len(target_imgs)} images...')
                for img_name in target_imgs: 
                    out_name = self.utility.replace_img_tag(img_name=img_name, tag=RDDBESRGAN)
                    real_esrgan = RealESRGAN(device='cuda', model_path=MODEL_RRDB_ESRGAN_X4)
                    real_esrgan.run(
                        input=f'{target["src_dir"]}/{img_name}',
                        output=f'{RES_DIR}/{out_name}'
                    )
                    torch.cuda.empty_cache()

                    # Print the progress
                    if checkpoint != 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1
                
                print(f'\t\tSuccessfully upscaled images in {self.utility.get_time_taken_str(start_time)}')

    def run_rrdb_psnr(self):
        print('\n=> Running RRDB (PSNR) on test datasets...')

        for step, target in enumerate(TARGETS_RRDB):
            print(f'\n\t[{step+1}/{len(TARGETS_RRDB)}] Running RRDB (PSNR) (X4) on {target["dataset"]} dataset...')

            RES_DIR = target['res_dir'] + '/RRDB_PSNR'

            print(f'\t\tSource directory: {target["src_dir"]}')
            print(f'\t\tResults directory: {RES_DIR}')

            self.utility.check_and_create_dir(RES_DIR)

            target_imgs = self.utility.get_imgs_with_tag_from_dir(dir_path=target['src_dir'], tag=target['src_tag'])[:10]

            result_imgs = self.utility.get_files_in_dir(RES_DIR)

            target_imgs = self.utility.filter_names_ignore_tag(src=target_imgs, res=result_imgs)

            if len(target_imgs) == 0:
                print('\t\tAlready upscaled images. Skipping this step.')
            else:
                start_time = time.time()
                idx = 1
                checkpoint = int(len(target_imgs)/PROGRESS_NUM)

                print(f'\t\tUpscaling {len(target_imgs)} images...')
                for img_name in target_imgs: 
                    out_name = self.utility.replace_img_tag(img_name=img_name, tag=RDDBPSNR)
                    real_esrgan = RealESRGAN(device='cuda', model_path=MODEL_RRDB_PSNR_X4)
                    real_esrgan.run(
                        input=f'{target["src_dir"]}/{img_name}',
                        output=f'{RES_DIR}/{out_name}'
                    )
                    torch.cuda.empty_cache()

                    # Print the progress
                    if checkpoint != 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1
                
                print(f'\t\tSuccessfully upscaled images in {self.utility.get_time_taken_str(start_time)}')
    
