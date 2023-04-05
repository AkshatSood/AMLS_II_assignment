"""Run Models Module"""

import time

import torch
import numpy as np
import PIL.Image as pil_image
import torch.backends.cudnn as cudnn

from constants2 import (MODEL_RRDB_ESRGAN_X4, MODEL_RRDB_PSNR_X4, PROGRESS_NUM,
                        RDDBESRGAN, RDDBPSNR, TARGETS_RRDB, RRDB_ESRGAN_DIR, RRDB_PSNR_DIR, TARGETS_FSRCNN)
from helpers.utility import Utility
from helpers.helpers import Helpers
from modules.real_esrgan import RealESRGAN
from modules.runner import Runner
from modules.FSRCNN import FSRCNN


class RunModels:

    def __init__(self):
        self.utility = Utility()
        self.helpers = Helpers()
        self.runner = Runner()

    def run_rrdb_esrgan_model(self):
        print('\n=> Running RRDB (ESRGAN) on test datasets...')

        for step, target in enumerate(TARGETS_RRDB):
            print(f'\n\t[{step+1}/{len(TARGETS_RRDB)}] Running RRDB (ESRGAN) (X4) on {target["dataset"]} dataset...')

            RES_DIR = target['res_dir'] + RRDB_ESRGAN_DIR

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

    def run_rrdb_psnr_model(self):
        print('\n=> Running RRDB (PSNR) on test datasets...')

        for step, target in enumerate(TARGETS_RRDB):
            print(f'\n\t[{step+1}/{len(TARGETS_RRDB)}] Running RRDB (PSNR) (X4) on {target["dataset"]} dataset...')

            RES_DIR = target['res_dir'] + RRDB_PSNR_DIR

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
    
    def run_fsrcnn_model(self):
        print('\n=> Running FSRCNN Model on test datasets...')

        for step, target in enumerate(TARGETS_FSRCNN):
            print(f'\n\t[{step+1}/{len(TARGETS_FSRCNN)}] Running FSRCNN (X{target["scale"]}) on {target["dataset"]} dataset...')

            print(f'\t\tSource directory: {target["src_dir"]}')
            print(f'\t\tResults directory: {target["res_dir"]}')

            self.utility.check_and_create_dir(target['res_dir'])

            target_imgs = self.utility.get_imgs_with_tag_from_dir(dir_path=target['src_dir'], tag=target['src_tag'])[:10]

            result_imgs = self.utility.get_files_in_dir(target['res_dir'])

            target_imgs = self.utility.filter_names_ignore_tag(src=target_imgs, res=result_imgs)

            if len(target_imgs) == 0:
                print('\t\tAlready upscaled images. Skipping this step.')
            else:
                start_time = time.time()
                idx = 1
                checkpoint = int(len(target_imgs)/PROGRESS_NUM)

                cudnn.benchmark = True
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

                fsrcnn = FSRCNN(scale_factor=target['scale']).to(device)

                state_dict = fsrcnn.state_dict()
                for n, p in torch.load(target['weights'], map_location=lambda storage, loc: storage).items():
                    if n in state_dict.keys():
                        state_dict[n].copy_(p)
                    else:
                        raise KeyError(n)

                fsrcnn.eval()

                print(f'\t\tUpscaling {len(target_imgs)} images...')
                for img_name in target_imgs: 
                    out_name = self.utility.replace_img_tag(img_name=img_name, tag='FSRCNN')
                    
                    img = pil_image.open(f'{target["src_dir"]}/{img_name}').convert('RGB')

                    # Increase the size of the image using bicubic interpolation
                    bicubic_img = img.resize(
                        (img.width * target['scale'],
                         img.height * target['scale']),
                        resample=pil_image.BICUBIC)

                    img, _ = self.helpers.preprocess_for_fsrcnn(img, device)
                    _, ycbcr_img = self.helpers.preprocess_for_fsrcnn(bicubic_img, device)

                    with torch.no_grad():
                        pred_img = fsrcnn(img).clamp(0.0, 1.0)

                    pred_img = pred_img.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

                    output_img = np.array([pred_img, ycbcr_img[..., 1], ycbcr_img[..., 2]]).transpose([1, 2, 0])
                    output_img = np.clip(self.helpers.convert_ycbcr_to_rgb(output_img), 0.0, 255.0).astype(np.uint8)
                    output_img = pil_image.fromarray(output_img)

                    output_img.save(f'{target["res_dir"]}/{out_name}')

                    # Print the progress
                    if checkpoint != 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1
                
                print(f'\t\tSuccessfully upscaled images in {self.utility.get_time_taken_str(start_time)}')
    