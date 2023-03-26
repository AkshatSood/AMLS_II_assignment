import cv2
import time

from constants import DATA_PROCESSING_TARGETS, PROCESSED_HR_SHAPE
from helpers.utility import Utility


class DataProcessing():
    
    def __init__(self):
        self.utility = Utility()
        self.hr_shape = PROCESSED_HR_SHAPE

        for target in DATA_PROCESSING_TARGETS: 
            self.utility.check_and_create_dir(target['HR_dir'])
            self.utility.check_and_create_dir(target['X2_dir'])
            self.utility.check_and_create_dir(target['X4_dir'])

    def process(self):
        """_summary_
        """
        
        for target in DATA_PROCESSING_TARGETS: 

            print(f'\tProcessing {target["name"]}...')
            
            img_names = self.utility.get_files_in_dir(target['raw_dir'])
            check_len = len(img_names)/10

            start_time = time.time()
            idx = 1

            print(f'\t\tProcessing {len(img_names)} files...')
            for img_name in img_names: 
                
                img = cv2.imread(f'{target["raw_dir"]}/{img_name}')

                center = img.shape
                x = center[1]/2 - self.hr_shape[1]/2
                y = center[0]/2 - self.hr_shape[0]/2

                crop_img = img[int(y):int(y+self.hr_shape[0]), int(x):int(x+self.hr_shape[1])]

                cv2.imwrite(f'{target["HR_dir"]}/{img_name}', crop_img)

                x2_img = cv2.resize(crop_img, (int(self.hr_shape[0]/2), int(self.hr_shape[1]/2)))
                cv2.imwrite(f'{target["X2_dir"]}/{img_name}', x2_img)

                x4_img = cv2.resize(crop_img, (int(self.hr_shape[0]/4), int(self.hr_shape[1]/4)))
                cv2.imwrite(f'{target["X4_dir"]}/{img_name}', x4_img)

                if (idx) % check_len == 0:
                    self.utility.progress_print(len(img_names), idx, start_time)
                idx += 1
