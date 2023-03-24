""" Utility Module """
import os
from pathlib import Path

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