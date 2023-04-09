"""Dataset Module"""

import h5py
import numpy as np
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from torch.utils.data import Dataset
from helpers.utility import Utility
from constants import DATASET_DIR, DATASET_DOWNLOAD_TARGETS

class Dataset:

    def __init__(self): 
        self.utility = Utility()
        self.utility.check_and_create_dir(DATASET_DIR)

    def download(self):
        print('\n=> Downloading and extracting datasets...')
        
        for target in DATASET_DOWNLOAD_TARGETS: 
            print(f'\tDownloading {target["name"]} dataset...')

            if self.utility.dir_exists(target['target_dir']):
                print(f'\t\tIt appears that this has already been downloaded at {target["target_dir"]}. Skipping this step.')
                continue

            with urlopen(target['url']) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(DATASET_DIR)

            print(f'\t\tSuccessfully downloaded dataset at {target["target_dir"]}.')

class FSRCNNTrainData(Dataset):
    def __init__(self, h5_file):
        super(FSRCNNTrainData, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
        
class FSRCNNValidationDataset(Dataset):
    def __init__(self, h5_file):
        super(FSRCNNValidationDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])