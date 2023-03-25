"""Track1 Module"""
import time
import cv2

from modules.bicubic import BicubicInterpolation
from helpers.utility import Utility
from constants import TRACK1_BICUBIC_TARGETS

class Track1:

    def __init__(self):
        self.utility = Utility()
        self.bicubic_interpolation = BicubicInterpolation()

        for target in TRACK1_BICUBIC_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

    def __run_bicubic_interpolation(self):
        print("Track 1 Bicubic Interpolation")

        for target in TRACK1_BICUBIC_TARGETS:
            print(f'\t{target["name"]}')

            start_time = time.time()
            idx = 1
            target_imgs = self.utility.get_files_in_dir(target['test_dir'])

            # Filter the images that have already been processed
            result_imgs = self.utility.get_files_in_dir(target['results_dir'])
            target_imgs = [img for img in target_imgs if img not in result_imgs]

            print(f'\tProcessing {len(target_imgs)} images...')

            for img_name in target_imgs:

                self.bicubic_interpolation.run(
                    input = f'{target["test_dir"]}/{img_name}',
                    output = f'{target["results_dir"]}/{img_name}',
                    scale = target['scale']
                )

                if (idx) % 10 == 0:
                    self.utility.progress_print(len(target_imgs), idx, start_time)
                idx+=1

    def run(self):
        self.__run_bicubic_interpolation()

