
from constants import CUSTOM_ESRGANX4_TARGETS
from helpers.utility import Utility
from modules.runner import Runner

class Custom:

    def __init__(self):
        self.utility = Utility()
        self.runner = Runner()

    def __run_real_esrgan(self): 
        print("Custom - Real-ESRGAN")

        for target in CUSTOM_ESRGANX4_TARGETS:
            self.utility.check_and_create_dir(target['results_dir'])

        self.runner.run_real_esrgan(CUSTOM_ESRGANX4_TARGETS)

    def run(self, run_real_esrgan=True): 
        if run_real_esrgan:
            self.__run_real_esrgan()



