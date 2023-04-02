"""Evaluation Module"""

import time

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
import imquality.brisque as brisque

from constants import (EVALUATION_DIR, EVALUATION_IMG_SHAPE,
                       EVALUATION_TARGETS, PROGRESS_NUM)
from helpers.helpers import Helpers
from helpers.utility import Utility


class Evaluation:

    def __init__(self):
        self.utility = Utility()
        self.helpers = Helpers()

        self.utility.check_and_create_dir(EVALUATION_DIR)

    def __compute_mse_and_psnr(self, hr_img, up_img):
        mse = np.sum((hr_img.astype("float") - up_img.astype("float")) ** 2)
        mse = mse / float(hr_img.shape[0] * hr_img.shape[1]) 

        psnr = 10 * np.log10(1.0 / mse)

        return mse, psnr

    def __compute_ssim(self, hr_img, up_img):
        return structural_similarity(hr_img, up_img)
        
    def __compute_brisque(self, img): 
        return brisque.score(img)
    
    def evaluate(self):
        
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
            hr_img_names = [hr_img_name for hr_img_name in hr_img_names if f'{hr_img_name.split(".")[0]}x{target["scale"]}.png' in up_img_names]

            # Check to make sure that the file lengths are the same and not zero
            if len(hr_img_names) == 0 or len(up_img_names) == 0: 
                print('\t\t\tError! No images found in either the HR or UP folders!')
                continue
            if len(hr_img_names) != len(up_img_names): 
                print('\t\t\tError! Different number of files were found for HR and UP images!')
            
            metrics = []
            idx = 1
            start_time = time.time()
            checkpoint = int(len(hr_img_names)/PROGRESS_NUM)

            for hr_img_name in hr_img_names: 
                up_img_name = f'{hr_img_name.split(".")[0]}x{target["scale"]}.png'
                
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

                # Compute the metrics
                ssim = self.__compute_ssim(hr_img_y, up_img_y)
                mse, psnr = self.__compute_mse_and_psnr(hr_img_y, up_img_y)
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
            



                
                    

            
