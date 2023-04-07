"""DataProcessing Module"""

import time

import cv2
import h5py
import numpy as np

from constants import DATA_PROCESSING_TARGETS, PROCESSED_HR_SHAPE, PROGRESS_NUM
from helpers.utility import Utility


class DataProcessing():

    def __init__(self):
        self.utility = Utility()
        self.hr_shape = PROCESSED_HR_SHAPE

        # Create the required directories if they do not already exist
        for target in DATA_PROCESSING_TARGETS:
            self.utility.check_and_create_dir(target['output_dir'])

    def __crop(self):
        """Reads the HR, X2 and X4 images provided in the dataset, crops 
        them into a smaller shape, and saves the new images in the 
        processed folder.  
        """
        print('\n=>Cropping images...')

        for step, target in enumerate(DATA_PROCESSING_TARGETS):
            print(f'\n\t[{step+1}/{len(DATA_PROCESSING_TARGETS)}] Cropping {target["name"]}...')
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

    def __create_dataset(self): 
        print('\n=> Creating datasets...')

        for step, target in enumerate(DATA_PROCESSING_TARGETS):
            print(f'\n\t[{step+1}/{len(DATA_PROCESSING_TARGETS)}] Creating dataset for {target["name"]}...')
            print(f'\t\tSource directory: {target["src_dir"]}')

            # If the .h5 file already exists, then skip to the next one
            if self.utility.file_exists(target['h5']):
                print(f'\t\t{target["h5"]} already exists. Skipping this step.')
                continue

            img_names = self.utility.get_files_in_dir(target['output_dir'])

            dim_x = int(self.hr_shape[1]/target['scale'])
            dim_y = int(self.hr_shape[0]/target['scale'])
            img_shape = (len(img_names), dim_y, dim_x, 3)

            h5f = h5py.File(target['h5'], 'w')
            h5f.create_dataset(
                name=target['name'],
                shape=img_shape, 
                maxshape=img_shape,
                dtype=np.int8)

            for idx, img_name in enumerate(img_names):
                img = cv2.imread(target['output_dir'] + '/' + img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                h5f[target['name']][idx, ...] = img[None]

            h5f.close()

            print(f'\t\tSuccessfully created dataset at {target["h5"]}')

    def process(self): 
        # self.__crop()
        self.__create_dataset()