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
        return sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

    def get_img_tag(self, img): 
        """Get the tag (eg HR, LR, etc) from the image name

        Args:
            img (str): Image name

        Returns:
            str: tag
        """
        return img.split('_')[4].split('.')[0]
    
    def get_img_num(self, img):
        """Get the image number from the image name

        Args:
            img (str): Image name

        Returns:
            str: Image number
        """
        return img.split('_')[1]

    def get_imgs_with_tag_from_dir(self, dir_path, tag):
        """Return a list of .png files in the directory with the 
        specified tag in the name.

        Args:
            dir_path (str): Directory path
            tag (str): Image tag

        Returns:
            list: List of .png files in the directory
        """
        return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.png') and self.get_img_tag(f) == tag]

    def replace_img_tag(self, img_name, tag):
        """Replace the tag in the image name with the provided tag

        Args:
            img_name (str): Image name
            tag (str): Tag

        Returns:
            str: New image name
        """
        num = img_name.split('_')[1]
        scale = img_name.split('_')[3]
        return f'img_{num}_SRF_{scale}_{tag}.png'
    
    def get_time_taken_str(self, start_time):
        """Returns a formatted string of the time taken between the start time and the current time

        Args:
            start_time (float): Start time

        Returns:
            str: Formatted time difference between start time and current time
        """
        time_taken = time.time() - start_time
        return str(datetime.timedelta(seconds=int(time_taken)))
    
    def filter_names_ignore_tag(self, src, res):
        """Filter the src image names, ignoring the tags in the result image names

        Args:
            src (list): List of image names to filter from 
            res (list): List of images to remove from the src

        Returns:
            list: List of filtered image
        """
        filtered = []
        res = [res_img[:13] for res_img in res]
        for src_img in src: 
            if not src_img[:13] in res:
                filtered.append(src_img)

        return filtered

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
        """Shave the border from the image
        """
        img = img[border: -border, border: -border]
        return img
