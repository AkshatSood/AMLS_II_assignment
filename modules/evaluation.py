"""Evaluation Module"""

import time

import cv2
import imquality.brisque as brisque
import numpy as np
import pandas as pd
import PIL.Image
import torch
from piqa import MS_SSIM
from skimage.measure import shannon_entropy
from skimage.metrics import (mean_squared_error, peak_signal_noise_ratio,
                             structural_similarity)
from torchvision import transforms

from constants import (EVALUATION_DIR, EVALUATION_IMG_SHAPE,
                       EVALUATION_TARGETS, PROGRESS_NUM)
from constants2 import TARGETS_EVALUATION
from helpers.helpers import Helpers
from helpers.utility import Utility


class Evaluation:

    def __init__(self):
        self.utility = Utility()
        self.helpers = Helpers()
        self.totensor = transforms.ToTensor()

        self.utility.check_and_create_dir(EVALUATION_DIR)

    def __compute_psnr(self, hr_img, up_img):
        return peak_signal_noise_ratio(hr_img, up_img)

    def __compute_mse(self, hr_img, up_img):
        return mean_squared_error(hr_img, up_img)

    def __compute_ssim(self, hr_img, up_img):
        return structural_similarity(hr_img, up_img)
    
    def __compute_ms_ssim(self, hr_img, up_img):
        return MS_SSIM(up_img, hr_img)
        
    def __compute_brisque(self, img): 
        return brisque.score(img)

    def __compute_entropy(self, img):
        return shannon_entropy(img)

    def __get_brisque_score(self, img): 
        score = 0 

        try: 
            score = brisque.score(img)
        except: 
            score = 100
 
        return score 
    
    def __compute_brisque_score(self, hr_path, up_path):
        
        # Read the images
        hr = PIL.Image.open(hr_path)
        up = PIL.Image.open(up_path)

        # Convert the images to YCbCr and extract the Y channel
        hr_y, _, _ = hr.convert('YCbCr').split()
        up_y, _, _ = up.convert('YCbCr').split()

        # Calculate the brisque scores for the RGB images
        hr_brisque = self.__get_brisque_score(hr)
        up_brisque = self.__get_brisque_score(up)

        # Calculate the brisque scores for the Y channel images
        hr_y_brisque = self.__get_brisque_score(hr_y)
        up_y_brisque = self.__get_brisque_score(up_y)

        return hr_brisque, up_brisque, hr_y_brisque, up_y_brisque

    
    def evaluate_div2k(self):
        
        metrics_summary = []

        for target in EVALUATION_TARGETS: 
            name = f'{target["track"]} - {target["method"]} (X{target["scale"]})'
            print(f'\n\tEvaluating {name}...')
            print(f'\t\tHR directory: {target["hr_dir"]}')
            print(f'\t\tUP directory: {target["up_dir"]}')
            
            # Get all the files in HR and UP (upscaled) images directories
            hr_img_names = self.utility.get_files_in_dir(target['hr_dir'])
            up_img_names = self.utility.get_files_in_dir(target['up_dir'])

            # Filter the list of files in the directory so that the same images are considered for evaluation
            up_img_names = [img_name for img_name in up_img_names if img_name.startswith(target['startswith'])]
            if target['scale'] > 1:
                hr_img_names = [hr_img_name for hr_img_name in hr_img_names if f'{hr_img_name.split(".")[0]}x{target["scale"]}.png' in up_img_names]

            # Check to make sure that the file lengths are the same and not zero
            if len(hr_img_names) == 0 or len(up_img_names) == 0: 
                print('\t\t\tError! No images found in either the HR or UP folders!')
                continue
            if len(hr_img_names) != len(up_img_names): 
                print('\t\t\tError! Different number of files were found for HR and UP images!')
                continue
            
            metrics = []
            idx = 1
            start_time = time.time()
            checkpoint = int(len(hr_img_names)/PROGRESS_NUM)

            for hr_img_name in hr_img_names: 
                up_img_name = f'{hr_img_name.split(".")[0]}x{target["scale"]}.png' if target['scale'] > 1 else hr_img_name
                
                # Read the images
                hr_img = cv2.imread(f'{target["hr_dir"]}/{hr_img_name}') 
                up_img = cv2.imread(f'{target["up_dir"]}/{up_img_name}')

                # Crop the images (so that they are both the same size)
                hr_img = self.helpers.crop_to_shape(hr_img, EVALUATION_IMG_SHAPE)
                up_img = self.helpers.crop_to_shape(up_img, EVALUATION_IMG_SHAPE)          

                # Convert the images to YCbCr
                hr_img_ycrcb = self.helpers.convert_rgb_to_ycbcr(cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB))
                up_img_ycrcb = self.helpers.convert_rgb_to_ycbcr(cv2.cvtColor(up_img, cv2.COLOR_BGR2RGB))

                # Extract the Y channel for comparison
                hr_img_y = hr_img_ycrcb[:,:,0]
                up_img_y = up_img_ycrcb[:,:,0]
                
                ssim, mse, psnr, brisque_score = None, None, None, None 
                # Compute the metrics
                ssim = self.__compute_ssim(hr_img_y, up_img_y)
                if target['scale'] > 1: 
                    mse = self.__compute_mse(hr_img_y, up_img_y)
                    psne = self.__compute_psnr(hr_img_y, up_img_y)
                brisque_score = self.__compute_brisque(up_img)


                metrics.append({
                    'ssim': ssim, 
                    'mse': mse,
                    'psnr': psnr, 
                    'brisque': brisque_score,
                })

                # Print the progress so far
                if checkpoint != 0 and (idx) % checkpoint == 0:
                    self.utility.progress_print(len(hr_img_names), idx, start_time)
                idx += 1

            # Create a dataframe for the metrics for this target
            metrics_df = pd.DataFrame(metrics)

            metrics_summary.append({
                'track': target['track'], 
                'method': target['method'], 
                'scale': target['scale'],
                'imgs': len(hr_img_names),
                'ssim_avg': metrics_df['ssim'].mean(),
                'ssim_min': metrics_df['ssim'].min(), 
                'ssim_max': metrics_df['ssim'].max(),
                'mse_avg': metrics_df['mse'].mean(),
                'mse_min': metrics_df['mse'].min(), 
                'mse_max': metrics_df['mse'].max(),
                'psnr_avg': metrics_df['psnr'].mean(),
                'psnr_min': metrics_df['psnr'].min(), 
                'psnr_max': metrics_df['psnr'].max(), 
                'brisque_avg': metrics_df['brisque'].mean(),
                'brisque_min': metrics_df['brisque'].min(), 
                'brisque_max': metrics_df['brisque'].max(), 
            })

            # Store the data from the metrics of this target
            output_csv = f'{EVALUATION_DIR}/{name}.csv'
            metrics_df.to_csv(output_csv, index=False)
            print(f'\t\tSuccessfully stored evaluation results in {output_csv}.')

        # Create a dataframe with the summary of all targets and store it
        summary_df = pd.DataFrame(metrics_summary)
        summary_output_file = f'{EVALUATION_DIR}/Summary.csv'
        summary_df.to_csv(summary_output_file, index=False)
        print(f'\n\tSuccessfully print evaluation summary to {summary_output_file}.')
            

    def evaluate_tests(self):
        for target in TARGETS_EVALUATION:
            print(f'\n=> Evaluating {target["dataset"]} (X{target["scale"]}) images...')

            # Get a sorted list of all the high resolution images
            hr_imgs = self.utility.get_imgs_with_tag_from_dir(dir_path=target['hr_dir'], tag='HR')
            hr_imgs.sort(key = lambda x: self.utility.get_img_num(x))

            metrics = []
            for idx, model in enumerate(target['models']):
                print(f'\t[{idx+1}/{len(target["models"])}] Evaluating {model["tag"]} images...')

                # Check if the upscaled img directory exists 
                if not self.utility.dir_exists(model['up_dir']):
                    print(f'\t\tError! {model["up_dir"]} does not exist!')
                    continue
                
                # Get a sorted list of all the upscaled images
                up_imgs = self.utility.get_imgs_with_tag_from_dir(dir_path=model['up_dir'], tag=model['tag'])
                up_imgs.sort(key = lambda x: self.utility.get_img_num(x))

                if len(hr_imgs) == 0 or len(up_imgs) == 0: 
                    print('\t\tError! No images found in either the HR or UP folders!')
                    continue
                if len(hr_imgs) != len(up_imgs): 
                    print('\t\tError! Different number of files were found for HR and UP images!')
                    continue
                
                for hr_img_name, up_img_name in zip(hr_imgs, up_imgs):
                    
                    # Read the images
                    hr_img = cv2.cvtColor(cv2.imread(f'{target["hr_dir"]}/{hr_img_name}'), cv2.COLOR_BGR2RGB)
                    up_img = cv2.cvtColor(cv2.imread(f'{model["up_dir"]}/{up_img_name}'), cv2.COLOR_BGR2RGB)

                    # Convert the images to YCbCr
                    hr_img_ycrcb = cv2.cvtColor(hr_img, cv2.COLOR_RGB2YCrCb)
                    up_img_ycrcb = cv2.cvtColor(up_img, cv2.COLOR_RGB2YCrCb)

                    # Extract the Y channel for comparison
                    hr_img_y, _, _ = cv2.split(hr_img_ycrcb)
                    up_img_y, _, _ = cv2.split(up_img_ycrcb)

                    rgb_mse = self.__compute_mse(hr_img=hr_img, up_img=up_img)
                    rgb_psnr = self.__compute_psnr(hr_img=hr_img, up_img=up_img)
                    ssim = self.__compute_ssim(hr_img=hr_img_y, up_img=up_img_y)
                    y_mse = self.__compute_mse(hr_img=hr_img_y, up_img=up_img_y)
                    y_psnr = self.__compute_psnr(hr_img=hr_img_y, up_img=up_img_y)

                    hr_brisque, up_brisque, hr_y_brisque, up_y_brisque = self.__compute_brisque_score(
                        hr_path=f'{target["hr_dir"]}/{hr_img_name}', 
                        up_path=f'{model["up_dir"]}/{up_img_name}'
                    )

                    brisque_perc = (up_brisque - hr_brisque)/hr_brisque
                    brisque_perc_y = (up_y_brisque - hr_y_brisque)/hr_y_brisque

                    hr_entropy = self.__compute_entropy(hr_img)
                    up_entropy = self.__compute_entropy(up_img)

                    entropy_perc = (up_entropy - hr_entropy) / hr_entropy

                    metrics.append({
                        'dataset': target['dataset'],
                        'scale': target['scale'],
                        'model': model['tag'],
                        'num': self.utility.get_img_num(hr_img_name),
                        'rgb_psnr': rgb_psnr,
                        'rgb_mse': rgb_mse,
                        'y_psnr': y_psnr, 
                        'y_mse': y_mse,
                        'ssim': ssim,
                        'hr_entropy': hr_entropy, 
                        'up_entropy': up_entropy,
                        'entropy_perc': entropy_perc,
                        'hr_brisque': hr_brisque, 
                        'up_brisque': up_brisque,
                        'hr_y_brisque': hr_y_brisque, 
                        'up_y_brisque': up_y_brisque,
                        'brisque_perc': brisque_perc,
                        'brisque_perc_y': brisque_perc_y
                    })
            
            metrics_df = pd.DataFrame(metrics)

            metrics_df.to_csv(target['eval_file'], index=False)

            print(f'\tSuccessfully stored evaluation in {target["eval_file"]}')

    def create_evaluation_summary(self):
        print('\n=> Creating evaluation summary')
        metrics_summary = []
        for idx, target in enumerate(TARGETS_EVALUATION):
            print(f'\t[{idx+1}/{len(TARGETS_EVALUATION)}] Creating summary for {target["dataset"]} (X{target["scale"]})...')
            
            if not self.utility.file_exists(target['eval_file']):
                print(f'\t\tUnable to find CSV file at {target["eval_file"]}. Skipping this dataset')
                continue
            
            metrics_df = pd.read_csv(target['eval_file'])

            for model in target['models']:
                metrics_summary.append({
                    'dataset': target['dataset'],
                    'scale': target['scale'],
                    'model': model['tag'],
                    'num': len(metrics_df.loc[metrics_df['model'] == model['tag']]),
                    'rgb_psnr_min': metrics_df.loc[metrics_df['model'] == model['tag']]['rgb_psnr'].min(),
                    'rgb_psnr_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['rgb_psnr'].mean(),
                    'rgb_psnr_max': metrics_df.loc[metrics_df['model'] == model['tag']]['rgb_psnr'].max(),
                    'rgb_mse_min': metrics_df.loc[metrics_df['model'] == model['tag']]['rgb_mse'].min(),
                    'rgb_mse_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['rgb_mse'].mean(),
                    'rgb_mse_max': metrics_df.loc[metrics_df['model'] == model['tag']]['rgb_mse'].max(),
                    'y_psnr_min': metrics_df.loc[metrics_df['model'] == model['tag']]['y_psnr'].min(), 
                    'y_psnr_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['y_psnr'].mean(), 
                    'y_psnr_max': metrics_df.loc[metrics_df['model'] == model['tag']]['y_psnr'].max(), 
                    'y_mse_min': metrics_df.loc[metrics_df['model'] == model['tag']]['y_mse'].min(),
                    'y_mse_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['y_mse'].mean(),
                    'y_mse_max': metrics_df.loc[metrics_df['model'] == model['tag']]['y_mse'].max(),
                    'ssim_min': metrics_df.loc[metrics_df['model'] == model['tag']]['ssim'].min(),
                    'ssim_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['ssim'].mean(),
                    'ssim_max': metrics_df.loc[metrics_df['model'] == model['tag']]['ssim'].max(),
                    'entropy_perc_min': metrics_df.loc[metrics_df['model'] == model['tag']]['entropy_perc'].min(),
                    'entropy_perc_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['entropy_perc'].mean(),
                    'entropy_perc_max': metrics_df.loc[metrics_df['model'] == model['tag']]['entropy_perc'].max(),
                    'brisque_perc_min': metrics_df.loc[metrics_df['model'] == model['tag']]['brisque_perc'].min(),
                    'brisque_perc_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['brisque_perc'].mean(),
                    'brisque_perc_max': metrics_df.loc[metrics_df['model'] == model['tag']]['brisque_perc'].max(),
                    'brisque_perc_y_min': metrics_df.loc[metrics_df['model'] == model['tag']]['brisque_perc_y'].min(),
                    'brisque_perc_y_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['brisque_perc_y'].mean(),
                    'brisque_perc_y_max': metrics_df.loc[metrics_df['model'] == model['tag']]['brisque_perc_y'].max(),
                    'hr_entropy_min': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_entropy'].min(), 
                    'hr_entropy_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_entropy'].min(), 
                    'hr_entropy_max': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_entropy'].min(), 
                    'up_entropy_min': metrics_df.loc[metrics_df['model'] == model['tag']]['up_entropy'].min(),
                    'up_entropy_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['up_entropy'].min(),
                    'up_entropy_max': metrics_df.loc[metrics_df['model'] == model['tag']]['up_entropy'].min(),
                    'hr_brisque_min': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_brisque'].min(), 
                    'hr_brisque_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_brisque'].min(), 
                    'hr_brisque_max': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_brisque'].min(), 
                    'up_brisque_min': metrics_df.loc[metrics_df['model'] == model['tag']]['up_brisque'].min(),
                    'up_brisque_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['up_brisque'].min(),
                    'up_brisque_max': metrics_df.loc[metrics_df['model'] == model['tag']]['up_brisque'].min(),
                    'hr_y_brisque_min': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_y_brisque'].min(), 
                    'hr_y_brisque_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_y_brisque'].min(), 
                    'hr_y_brisque_max': metrics_df.loc[metrics_df['model'] == model['tag']]['hr_y_brisque'].min(), 
                    'up_y_brisque_min': metrics_df.loc[metrics_df['model'] == model['tag']]['up_y_brisque'].min(),
                    'up_y_brisque_avg': metrics_df.loc[metrics_df['model'] == model['tag']]['up_y_brisque'].min(),
                    'up_y_brisque_max': metrics_df.loc[metrics_df['model'] == model['tag']]['up_y_brisque'].min(),
                })

        summary_df = pd.DataFrame(metrics_summary)
        summary_df.to_csv('./evaluation/Evaluation Summary.csv', index=False)

        print('\tSuccessfully stored evaluation summary in ./evaluation/Evaluation Summary.csv')


                


                
                    

            
