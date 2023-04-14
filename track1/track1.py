"""Track1 Module"""

from constants import (TRACK1_BICUBIC_TARGETS, TRACK1_ESRGANX4_TARGETS,
                       TRACK1_FSRCNN_TARGETS, TRACK1_SRCNN_TARGETS)
from helpers.utility import Utility
from modules.runner import Runner


class Track1:

    def __init__(self):
        self.utility = Utility()
        self.runner = Runner()

    def __run_bicubic_interpolation(self):
        print("Track 1 - Bicubic Interpolation")

        # Create the results directories if they do not already exist
        for target in TRACK1_BICUBIC_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_bicubic_interpolation(TRACK1_BICUBIC_TARGETS)

    def __run_esrgan(self):
        print("Track 1 - ESRGAN")

        # Create the results directories if they do not already exist
        for target in TRACK1_ESRGANX4_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_esrgan(TRACK1_ESRGANX4_TARGETS)

    def __run_srcnn(self):
        print('Track 1 - SRCNN')

        # Create the results directories if they do not already exist
        for target in TRACK1_SRCNN_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_srcnn(TRACK1_SRCNN_TARGETS)

    def __run_fsrcnn(self):
        print('Track 1 - FSRCNN')

        # Create the results directories if they do not already exist
        for target in TRACK1_FSRCNN_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_fsrcnn(TRACK1_FSRCNN_TARGETS)

    def run(self, run_bicubic_interpolation=True, run_esrgan=True, run_srcnn=True, run_fsrcnn=True):
        if run_esrgan:
            self.__run_esrgan()
        if run_bicubic_interpolation:
            self.__run_bicubic_interpolation()
        if run_srcnn:
            self.__run_srcnn()
        if run_fsrcnn:
            self.__run_fsrcnn()
