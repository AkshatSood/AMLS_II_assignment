"""DataProcessing Module"""

import time

import cv2
import h5py
import numpy as np

from constants import DATA_PROCESSING_CROP_TARGETS, PROCESSED_HR_SHAPE, PROGRESS_NUM, DATA_PROCESSING_H5_TARGETS
from helpers.utility import Utility
from helpers.helpers import Helpers

class DataProcessing():

    def __init__(self):
        """Default constructor
        """
        self.utility = Utility()
        self.helpers = Helpers()
        self.hr_shape = PROCESSED_HR_SHAPE

        # Create the required directories if they do not already exist
        for target in DATA_PROCESSING_CROP_TARGETS:
            self.utility.check_and_create_dir(target['output_dir'])

    def crop_images(self):
        """Reads the HR, X2 and X4 images provided in the dataset, crops 
        them into a smaller shape, and saves the new images in the 
        processed folder.  
        """
        print('\n=>Cropping images...')

        for step, target in enumerate(DATA_PROCESSING_CROP_TARGETS):
            print(f'\n\t[{step+1}/{len(DATA_PROCESSING_CROP_TARGETS)}] Cropping {target["name"]}...')
            print(f'\t\tSource directory: {target["src_dir"]}')

            img_names = self.utility.get_files_in_dir(target['src_dir'])
            processed_img_names = []
            if self.utility.dir_exists(target['output_dir']):
                processed_img_names = self.utility.get_files_in_dir(target['output_dir'])

            img_names = [name for name in img_names if not name in processed_img_names]

            check_len = int(len(img_names)/PROGRESS_NUM)

            dim_x = int(self.hr_shape[1]/target['scale'])
            dim_y = int(self.hr_shape[0]/target['scale'])

            start_time = time.time()
            idx = 1

            if len(img_names) == 0:
                print(f'\t\tAlready cropped images. Can be found in {target["output_dir"]}')
            else:
                print(f'\t\tProcessing {len(img_names)} files...')
                for img_name in img_names:

                    img = cv2.imread(f'{target["src_dir"]}/{img_name}')

                    center = img.shape

                    x = center[1]/2 - dim_x/2
                    y = center[0]/2 - dim_y/2

                    crop_img = img[int(y):int(y+dim_y), int(x):int(x+dim_x)]

                    cv2.imwrite(f'{target["output_dir"]}/{img_name}', crop_img)

                    if check_len != 0 and (idx) % check_len == 0:
                        self.utility.progress_print(len(img_names), idx, start_time)
                    idx += 1

                print(f'\t\tSuccessfully cropped the images. Can be found in {target["output_dir"]}')

    def create_training_datasets(self, hr_patch_size=324, hr_step=162): 
        """Create HDF5 binary data datasets (.h5 files) from the DIV2K image patches
        """
        print('\n=> Creating datasets...')

        for step, target in enumerate(DATA_PROCESSING_H5_TARGETS):
            print(f'\n\t[{step+1}/{len(DATA_PROCESSING_H5_TARGETS)}] Creating dataset for {target["name"]}...')
            print(f'\t\tLR directory: {target["lr_dir"]}')
            print(f'\t\tHR directory: {target["hr_dir"]}')

            # If the .h5 file already exists, then skip to the next one
            if self.utility.file_exists(target['h5']):
                print(f'\t\t{target["h5"]} already exists. Skipping this step.')
                continue

            lr_img_names = self.utility.get_files_in_dir(target['lr_dir'])
            hr_img_names = self.utility.get_files_in_dir(target['hr_dir'])

            if(len(lr_img_names) != len(hr_img_names)):
                print('\t\t\tError! Number of HR and LR images is not the same!')
                continue
            if len(hr_img_names) == 0 or len(lr_img_names) == 0: 
                print('\t\t\tError! Cannot find LR or HR images!')
                continue

            print(f'\t\tFound {len(hr_img_names)} images...')

            start_time = time.time()
            checkpoint = int(len(hr_img_names)/PROGRESS_NUM)

            if target['training']:
                
                lr_patches = []
                hr_patches = []

                scale = target['scale']
                patch_size = int(hr_patch_size/scale)
                step = int(hr_step/scale)

                for idx, (lr_img_name, hr_img_name) in enumerate(zip(lr_img_names, hr_img_names)):

                    lr_img = cv2.imread(target['lr_dir'] + '/' + lr_img_name)
                    hr_img = cv2.imread(target['hr_dir'] + '/' + hr_img_name)

                    lr = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                    hr = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

                    hr = np.array(hr).astype(np.float32)
                    lr = np.array(lr).astype(np.float32)

                    lr = self.helpers.convert_rgb_to_y(lr)
                    hr = self.helpers.convert_rgb_to_y(hr)

                    lr = np.asarray(lr).astype(np.float32)
                    hr = np.asarray(hr).astype(np.float32)
                    
                    for i in range(0, lr.shape[0] - patch_size + 1, step):
                        for j in range(0, lr.shape[1] - patch_size + 1, step):
                            lr_patches.append(lr[i:i+patch_size, j:j+patch_size])
                            hr_patches.append(hr[i*scale:i*scale+patch_size*scale, j*scale:j*scale+patch_size*scale])
                    
                    if checkpoint != 0 and (idx+1) % checkpoint == 0:
                        self.utility.progress_print(len(hr_img_names), idx+1, start_time)

                print(f'\t\tCreated training dataset with {len(hr_patches)} ({hr_patches[0].shape} HR and {lr_patches[0].shape} LR) patches.')
                
                hr_patches = np.asarray(hr_patches)
                lr_patches = np.asarray(lr_patches)

                h5f = h5py.File(target['h5'], 'w')

                lr_start_time = time.time()
                h5f.create_dataset('lr', data=lr_patches)
                print(f'\t\t\tCreated LR dataset in {self.utility.get_time_taken_str(lr_start_time)}')

                hr_start_time = time.time()
                h5f.create_dataset('hr', data=hr_patches)
                print(f'\t\t\tCreated HR dataset in {self.utility.get_time_taken_str(hr_start_time)}')

                h5f.close()

            else:
                h5f = h5py.File(target['h5'], 'w')
                lr_group = h5f.create_group('lr')
                hr_group = h5f.create_group('hr')

                for idx, (lr_img_name, hr_img_name) in enumerate(zip(lr_img_names, hr_img_names)):

                    lr_img = cv2.imread(target['lr_dir'] + '/' + lr_img_name)
                    hr_img = cv2.imread(target['hr_dir'] + '/' + hr_img_name)

                    lr = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                    hr = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

                    hr = np.array(hr).astype(np.float32)
                    lr = np.array(lr).astype(np.float32)

                    lr = self.helpers.convert_rgb_to_y(lr)
                    hr = self.helpers.convert_rgb_to_y(hr)

                    lr_group.create_dataset(str(idx), data=lr)
                    hr_group.create_dataset(str(idx), data=hr)

                    if checkpoint != 0 and (idx+1) % checkpoint == 0:
                        self.utility.progress_print(len(hr_img_names), idx+1, start_time)

                h5f.close()

            print(f'\t\tSuccessfully created dataset at {target["h5"]}')
