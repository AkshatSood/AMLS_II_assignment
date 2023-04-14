"""Track2 Module"""

from constants import (TRACK2_BICUBIC_TARGETS, TRACK2_ESRGANX4_TARGETS,
                       TRACK2_FSRCNN_TARGETS, TRACK2_SRCNN_TARGETS)
from helpers.utility import Utility
from modules.runner import Runner


class Track2:

    def __init__(self):
        self.utility = Utility()
        self.runner = Runner()

    def __run_bicubic_interpolation(self):
        print("Track 2 - Bicubic Interpolation")

        # Create the results directories if they do not already exist
        for target in TRACK2_BICUBIC_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_bicubic_interpolation(TRACK2_BICUBIC_TARGETS)

    def __run_esrgan(self):
        print("Track 2 - ESRGAN")

        # Create the results directories if they do not already exist
        for target in TRACK2_ESRGANX4_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_esrgan(TRACK2_ESRGANX4_TARGETS)

    def __run_srcnn(self):
        print('Track 2 - SRCNN')

        # Create the results directories if they do not already exist
        for target in TRACK2_SRCNN_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_srcnn(TRACK2_SRCNN_TARGETS)

    def __run_fsrcnn(self):
        print('Track 2 - FSRCNN')

        # Create the results directories if they do not already exist
        for target in TRACK2_FSRCNN_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_fsrcnn(TRACK2_FSRCNN_TARGETS)

    def run(self, run_bicubic_interpolation=True, run_esrgan=True, run_srcnn=True, run_fsrcnn=True):
        if run_esrgan:
            self.__run_esrgan()
        if run_bicubic_interpolation:
            self.__run_bicubic_interpolation()
        if run_srcnn:
            self.__run_srcnn()
        if run_fsrcnn:
            self.__run_fsrcnn()
