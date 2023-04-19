"""Run Models Module"""

import time

import numpy as np
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn

from constants2 import (MODEL_RRDB_ESRGAN_X4, MODEL_RRDB_PSNR_X4, PROGRESS_NUM,
                        RRDBESRGAN, RRDBPSNR, RRDB_ESRGAN_DIR, RRDB_PSNR_DIR,
                        TARGETS_FSRCNN, TARGETS_RRDB)
from helpers.helpers import Helpers
from helpers.utility import Utility
from modules.FSRCNN import FSRCNN
from modules.ESRGAN import ESRGAN

class RunModels:
    """Run the various models in the project
    """

    def __init__(self):
        self.utility = Utility()
        self.helpers = Helpers()

    def run_rrdb_esrgan_model(self):
        """Upscale the test dataset images with RRDBESRGAN model
        The code provided at https://github.com/xinntao/ESRGAN
        has been used as reference
        """
        print('\n=> Running RRDB (ESRGAN) on test datasets...')
        
        # Loop over all the targets
        for step, target in enumerate(TARGETS_RRDB):
            print(f'\n\t[{step+1}/{len(TARGETS_RRDB)}] Running RRDB (ESRGAN) (X4) on {target["dataset"]} dataset...')

            RES_DIR = target['res_dir'] + RRDB_ESRGAN_DIR

            print(f'\t\tSource directory: {target["src_dir"]}')
            print(f'\t\tResults directory: {RES_DIR}')

            self.utility.check_and_create_dir(RES_DIR)

            # Get a list of target images and filter the ones that have already been upscaled
            target_imgs = self.utility.get_imgs_with_tag_from_dir(dir_path=target['src_dir'], tag=target['src_tag'])
            result_imgs = self.utility.get_files_in_dir(RES_DIR)
            target_imgs = self.utility.filter_names_ignore_tag(src=target_imgs, res=result_imgs)

            if len(target_imgs) == 0:
                print('\t\tAlready upscaled images. Skipping this step.')
            else:
                # Keep track of the progress
                start_time = time.time()
                idx = 1
                checkpoint = int(len(target_imgs)/PROGRESS_NUM)

                # Upscale each image
                print(f'\t\tUpscaling {len(target_imgs)} images...')
                for img_name in target_imgs: 
                    out_name = self.utility.replace_img_tag(img_name=img_name, tag=RRDBESRGAN)
                    model = ESRGAN(device='cuda', model_path=MODEL_RRDB_ESRGAN_X4)
                    model.run(
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
        """Upscale the test dataset images with the RRDBPSNR model
        The code provided at https://github.com/xinntao/ESRGAN
        has been used as reference
        """
        print('\n=> Running RRDB (PSNR) on test datasets...')

        # Loop over all the targets
        for step, target in enumerate(TARGETS_RRDB):
            print(f'\n\t[{step+1}/{len(TARGETS_RRDB)}] Running RRDB (PSNR) (X4) on {target["dataset"]} dataset...')

            RES_DIR = target['res_dir'] + RRDB_PSNR_DIR

            print(f'\t\tSource directory: {target["src_dir"]}')
            print(f'\t\tResults directory: {RES_DIR}')

            self.utility.check_and_create_dir(RES_DIR)

            # Get a list of all the target images, and filter the ones that have already been upscaled
            target_imgs = self.utility.get_imgs_with_tag_from_dir(dir_path=target['src_dir'], tag=target['src_tag'])
            result_imgs = self.utility.get_files_in_dir(RES_DIR)
            target_imgs = self.utility.filter_names_ignore_tag(src=target_imgs, res=result_imgs)

            if len(target_imgs) == 0:
                print('\t\tAlready upscaled images. Skipping this step.')
            else:
                start_time = time.time()
                idx = 1
                checkpoint = int(len(target_imgs)/PROGRESS_NUM)

                # Loop over all the target images
                print(f'\t\tUpscaling {len(target_imgs)} images...')
                for img_name in target_imgs: 
                    out_name = self.utility.replace_img_tag(img_name=img_name, tag=RRDBPSNR)
                    model = ESRGAN(device='cuda', model_path=MODEL_RRDB_PSNR_X4)
                    model.run(
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
        """Upscale test dataset images using the FSRCNN models
        The code provided at https://github.com/yjn870/FSRCNN-pytorch
        has been used as reference
        """
        print('\n=> Running FSRCNN Model on test datasets...')

        for step, target in enumerate(TARGETS_FSRCNN):
            print(f'\n\t[{step+1}/{len(TARGETS_FSRCNN)}] Running {target["model"]} (X{target["scale"]}) on {target["dataset"]} dataset...')

            print(f'\t\tSource directory: {target["src_dir"]}')
            print(f'\t\tResults directory: {target["res_dir"]}')

            self.utility.check_and_create_dir(target['res_dir'])

            # Get a list of images to be upscaled and filter the ones that have already been upscaled
            target_imgs = self.utility.get_imgs_with_tag_from_dir(dir_path=target['src_dir'], tag=target['src_tag'])
            result_imgs = self.utility.get_files_in_dir(target['res_dir'])
            target_imgs = self.utility.filter_names_ignore_tag(src=target_imgs, res=result_imgs)

            if len(target_imgs) == 0:
                print('\t\tAlready upscaled images. Skipping this step.')
            else:
                # Keep track of the progress
                start_time = time.time()
                idx = 1
                checkpoint = int(len(target_imgs)/PROGRESS_NUM)

                cudnn.benchmark = True
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

                # Load the FSRCNN model, and initialize with the saved weights
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
                    out_name = self.utility.replace_img_tag(img_name=img_name, tag=target['model'])
                    
                    img = pil_image.open(f'{target["src_dir"]}/{img_name}').convert('RGB')

                    # Increase the size of the image using bicubic interpolation
                    bicubic_img = img.resize(
                        (img.width * target['scale'],
                         img.height * target['scale']),
                        resample=pil_image.BICUBIC)

                    # Get the normalised YCBCR colour space image
                    img, _ = self.helpers.preprocess_for_fsrcnn(img, device)
                    # Get the YCBCR colour space image
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
    