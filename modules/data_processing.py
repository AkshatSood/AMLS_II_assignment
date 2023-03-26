"""DataProcessing Module"""
import cv2
import time

from constants import DATA_PROCESSING_TARGETS, PROCESSED_HR_SHAPE
from helpers.utility import Utility


class DataProcessing():
    
    def __init__(self):
        self.utility = Utility()
        self.hr_shape = PROCESSED_HR_SHAPE

        # Create the required directories if they do not already exist
        for target in DATA_PROCESSING_TARGETS: 
            self.utility.check_and_create_dir(target['output_dir'])

    def process(self):
        """Reads the HR, X2 and X4 images provided in the dataset, crops 
        them into a smaller shape, and saves the new images in the 
        processed folder.  
        """
        
        for target in DATA_PROCESSING_TARGETS: 

            print(f'\tProcessing {target["name"]}...')
            
            img_names = self.utility.get_files_in_dir(target['raw_dir'])
            processed_img_names = []
            if self.utility.dir_exists(target['output_dir']):
                processed_img_names = self.utility.get_files_in_dir(target['output_dir'])
            
            img_names = [name for name in img_names if not name in processed_img_names]

            check_len = int(len(img_names)/10)

            dim_x = int(self.hr_shape[1]/target['scale'])
            dim_y = int(self.hr_shape[0]/target['scale'])

            start_time = time.time()
            idx = 1

            print(f'\t\tProcessing {len(img_names)} files...')
            for img_name in img_names: 
                
                img = cv2.imread(f'{target["raw_dir"]}/{img_name}')

                center = img.shape

                x = center[1]/2 - dim_x/2
                y = center[0]/2 - dim_y/2

                crop_img = img[int(y):int(y+dim_y), int(x):int(x+dim_x)]

                cv2.imwrite(f'{target["output_dir"]}/{img_name}', crop_img)

                if (idx) % check_len == 0:
                    self.utility.progress_print(len(img_names), idx, start_time)
                idx += 1

            print(f'\t\tSuccessfully processed the images. Can be found in {target["output_dir"]}')