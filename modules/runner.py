"""Runner Module"""

import time

import cv2
import numpy as np
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn

from constants import PROGRESS_NUM
from helpers.helpers import Helpers
from helpers.utility import Utility
from modules.bicubic import BicubicInterpolation
from modules.FSRCNN import FSRCNN
from modules.real_esrgan import RealESRGAN
from modules.SRCNN import SRCNN


class Runner:

    def __init__(self):
        self.utility = Utility()
        self.helpers = Helpers()

    def run_bicubic_interpolation(self, targets):
        """Run bicubic interpolation in the image files provided

        Args:
            targets (list): list of directories with images
        """

        bicubic_interpolation = BicubicInterpolation()

        # Loop over all the image directories
        for target in targets:
            print(f'\t{target["name"]}')
            print(f'\t\tSource directory: {target["src_dir"]}')

            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['src_dir'])

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
                        input=f'{target["src_dir"]}/{img_name}',
                        output=f'{target["results_dir"]}/{img_name}',
                        scale=target['scale']
                    )

                    if checkpoint != 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1

                print(f'\t\tSuccessfully upscaled images. Can be found in {target["results_dir"]}')

    def run_real_esrgan(self, targets):
        for target in targets:
            print(f'\t{target["name"]}')
            print(f'\t\tSource directory: {target["src_dir"]}')

            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['src_dir'])

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
                    real_esrgan = RealESRGAN(device='cuda', model=target['model'])
                    real_esrgan.run(
                        input=f'{target["src_dir"]}/{img_name}',
                        output=f'{target["results_dir"]}/{img_name}'
                    )
                    torch.cuda.empty_cache()

                    # Print the progress
                    if checkpoint != 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1

                print(f'\t\tSuccessfully upscaled images. Can be found in {target["results_dir"]}')

    def run_srcnn(self, targets):
        for target in targets:
            print(f'\t{target["name"]}')
            print(f'\t\tSource directory: {target["src_dir"]}')

            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['src_dir'])

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
                    img = cv2.imread(f'{target["src_dir"]}/{img_name}')

                    print(img.shape)
                    img = self.utility.mod_crop(img, target['scale'])

                    y_cr_cb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

                    test_img = np.zeros((1, y_cr_cb_img.shape[0], y_cr_cb_img.shape[1], 1), dtype=float)
                    test_img[0, :, :, 0] = y_cr_cb_img[:, :, 0].astype(float) / 255

                    pred_img = srcnn.predict(test_img)

                    pred_img *= 255
                    pred_img[pred_img[:] > 255] = 255
                    pred_img[pred_img[:] < 0] = 0
                    pred_img = pred_img.astype(np.uint8)

                    y_cr_cb_img = self.utility.shave(y_cr_cb_img, target['border'])
                    y_cr_cb_img[:, :, 0] = pred_img[0, :, :, 0]

                    pred_img = cv2.cvtColor(y_cr_cb_img, cv2.COLOR_YCrCb2BGR)

                    cv2.imwrite(f'{target["results_dir"]}/{img_name}', pred_img)

                    # Print the progress
                    if checkpoint != 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1

                print(f'\t\tSuccessfully processed images. Can be found in {target["results_dir"]}')

    def run_fsrcnn(self, targets):
        cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for target in targets:
            print(f'\t{target["name"]}')
            print(f'\t\tSource directory: {target["src_dir"]}')


            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['src_dir'])

            # Filter the images that have already been processed
            result_imgs = self.utility.get_files_in_dir(target['results_dir'])
            target_imgs = [img for img in target_imgs if img not in result_imgs]

            checkpoint = int(len(target_imgs)/PROGRESS_NUM)

            # If all the images have been processed, then skip
            if len(target_imgs) == 0:
                print(f'\t\tAlready upscaled images. Can be found in {target["results_dir"]}')
            else:
                print(f'\t\tProcessing {len(target_imgs)} images...')

                fsrcnn = FSRCNN(scale_factor=target['scale']).to(device)

                state_dict = fsrcnn.state_dict()
                for n, p in torch.load(target['weights_file'], map_location=lambda storage, loc: storage).items():
                    if n in state_dict.keys():
                        state_dict[n].copy_(p)
                    else:
                        raise KeyError(n)

                fsrcnn.eval()

                for img_name in target_imgs:

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

                    output_img.save(f'{target["results_dir"]}/{img_name}')

                    # Print the progress
                    if checkpoint != 0 and (idx) % checkpoint == 0:
                        self.utility.progress_print(len(target_imgs), idx, start_time)
                    idx += 1

                print(f'\t\tSuccessfully upscaled images. Can be found in {target["results_dir"]}')
