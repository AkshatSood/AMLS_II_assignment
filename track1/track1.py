"""Track1 Module"""

from constants import TRACK1_BICUBIC_TARGETS, TRACK1_ESRGANX4_TARGETS
from helpers.utility import Utility
from modules.runner import Runner


class Track1:

    def __init__(self):
        self.utility = Utility()
        self.runner = Runner()

    def __run_bicubic_interpolation(self):
        print("Track 1 - Bicubic Interpolation")

        for target in TRACK1_BICUBIC_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_bicubic_interpolation(TRACK1_BICUBIC_TARGETS)

    def __run_real_esrgan(self): 
        print("Track 1 - Real-ESRGAN")

        for target in TRACK1_ESRGANX4_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_real_esrgan(TRACK1_ESRGANX4_TARGETS)

    def run(self, run_bicubic_interpolation=True, run_real_esrgan=True):
        
        if run_bicubic_interpolation:
            self.__run_bicubic_interpolation()
        if run_real_esrgan:
            self.__run_real_esrgan()
