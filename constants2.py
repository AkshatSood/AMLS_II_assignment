"""Test Constants Module"""

# TEST DATASET NAMES
SET5 = 'Set5'
SET14 = 'Set14'
URBAN100 = 'Urban100'
BSD100 = 'BSD100'
SUNHAYS80 = 'SunHays80'

# IMG_TAGS
HR = 'HR'
LR = 'LR'
BICUBIC = 'bicubic'
GLASNER = 'glasner'
NEAREST = 'nearest'
SRCNN = 'SRCNN'
RDDBESRGAN = 'RDDBESRGAN'
RDDBPSNR = 'RDDBPSNR'

# DATASET ROOTS 
DATASET_DIR = './dataset'
SET5_ROOT = DATASET_DIR + '/' + SET5
SET14_ROOT = DATASET_DIR + '/' + SET14
URBAN100_ROOT = DATASET_DIR + '/' + URBAN100
BSD100_ROOT = DATASET_DIR + '/' + BSD100
SUNHAYS80_ROOT = DATASET_DIR + '/' + SUNHAYS80

# DATASET SUB PATHS
IMGAGE_SRF_X2 = '/image_SRF_2'
IMGAGE_SRF_X3 = '/image_SRF_3'
IMGAGE_SRF_X4 = '/image_SRF_4'
IMGAGE_SRF_X8 = '/image_SRF_8'

# MODELS
MODELS_DIR = './models'
MODEL_RRDB_ESRGAN_X4 = MODELS_DIR + '/RRDB_ESRGAN_x4.pth'
MODEL_RRDB_PSNR_X4 = MODELS_DIR + '/RRDB_PSNR_x4.pth'

# RESULTS ROOTS
RESULTS_DIR = './results'
SET5_RESULTS_ROOT = RESULTS_DIR + '/' + SET5
SET14_RESULTS_ROOT = RESULTS_DIR + '/' + SET14
URBAN100_RESULTS_ROOT = RESULTS_DIR + '/' + URBAN100
BSD100_RESULTS_ROOT = RESULTS_DIR + '/' + BSD100
SUNHAYS80_RESULTS_ROOT = RESULTS_DIR + '/' + SUNHAYS80

# RESULTS SUB PATHS
RRDB_ESRGAN = '/RRDB_ESRGAN'
RRDB_PSNR = '/RRDB_PSNR'

SCALE_X2 = 2 
SCALE_X3 = 3 
SCALE_X4 = 4 
SCALE_X8 = 8
PROGRESS_NUM = 10

TARGETS_RRDB = [
    {
        'dataset': SET5, 
        'src_dir': SET5_ROOT + IMGAGE_SRF_X4,
        'res_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4,
        'src_tag': LR
    },
    {
        'dataset': SET14, 
        'src_dir': SET14_ROOT + IMGAGE_SRF_X4,
        'res_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4, 
        'src_tag': LR
    },
    {
        'dataset': URBAN100, 
        'src_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
        'res_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4, 
        'src_tag': LR
    },
    {
        'dataset': BSD100, 
        'src_dir': BSD100_ROOT + IMGAGE_SRF_X4,
        'res_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4, 
        'src_tag': LR
    }
]

