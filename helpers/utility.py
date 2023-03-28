""" Utility Module """

import datetime
import os
import time
from pathlib import Path

import numpy as np


class Utility:
    """Provides common functions used across the project
    """

    def file_exists(self, fname):
        """Checks if the file exists, and is a file

        Args:
            fname (str): file path

        Returns:
            bool: True if the file exists, False otherwise
        """
        file_path = Path(fname)
        return file_path.is_file()

    def dir_exists(self, dir_path):
        """Checks if the directory exists

        Args:
            dir_path (str): path of the directory

        Returns:
            bool: True if the directory exists, False otherwise
        """
        return os.path.exists(dir_path)

    def check_and_create_dir(self, dir_path):
        """Check if the dir exists, if not, then create it

        Args:
            dir_path (str): path of the directory
        """
        if not self.dir_exists(dir_path):
            os.makedirs(dir_path)

    def get_files_in_dir(self, dir_path):
        """Get list of files in directory

        Args:
            dir_path (str): path of directory
        """
        return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    def progress_print(self, total_itrs, completed_itrs, start_time):
        """Prints a progress message for the process, based on progress
        so far. Computes an expected time for completion by calculating
        average time per completed iteration and number of iterations left.

        Args:
            total_itrs (int): total number of iterations
            completed_itrs (int): number of iterations completed so far
            start_time (time): start time of the process
        """
        time_taken = time.time() - start_time
        time_taken_str = str(datetime.timedelta(seconds=int(time_taken)))

        tpi = time_taken/completed_itrs

        time_left = (total_itrs - completed_itrs) * tpi
        time_left_str = str(datetime.timedelta(seconds=int(time_left)))

        print(f'\t\t\t{completed_itrs}/{total_itrs}\tTime Elapsed: {time_taken_str} (TPI: {tpi:.2f}s)\tTime Left: {time_left_str}')

    def mod_crop(self, img, scale):
        size = img.shape[0:2]
        size = size - np.mod(size, scale)
        img = img[0:size[0], 1:size[1]]
        return img

    def shave(self, img, border):
        img = img[border: -border, border: -border]
        return img
