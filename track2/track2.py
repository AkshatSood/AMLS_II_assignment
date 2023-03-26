"""Track2 Module"""

from constants import TRACK2_BICUBIC_TARGETS, TRACK2_ESRGANX4_TARGETS
from helpers.utility import Utility
from modules.runner import Runner

class Track2: 

    def __init__(self):
        self.utility = Utility()
        self.runner = Runner()

    def __run_bicubic_interpolation(self):
        print("Track 2 - Bicubic Interpolation")

        for target in TRACK2_BICUBIC_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_bicubic_interpolation(TRACK2_BICUBIC_TARGETS)

    def __run_real_esrgan(self): 
        print("Track 2 - Real-ESRGAN")

        for target in TRACK2_ESRGANX4_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_real_esrgan(TRACK2_ESRGANX4_TARGETS)

    def run(self, run_bicubic_interpolation=True, run_real_esrgan=True):
        
        if run_bicubic_interpolation:
            self.__run_bicubic_interpolation()
        if run_real_esrgan:
            self.__run_real_esrgan()