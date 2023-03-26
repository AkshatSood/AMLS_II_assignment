"""Constants Module"""

# DATASET PATHS
DATASET_DIR = './dataset'
TRACK1_X2_TRAIN = DATASET_DIR + '/DIV2K_train_LR_bicubic/X2'
TRACK1_X3_TRAIN = DATASET_DIR + '/DIV2K_train_LR_bicubic/X3'
TRACK1_X4_TRAIN = DATASET_DIR + '/DIV2K_train_LR_bicubic/X4'
TRACK2_X2_TRAIN = DATASET_DIR + '/DIV2K_train_LR_unknown/X2'
TRACK2_X3_TRAIN = DATASET_DIR + '/DIV2K_train_LR_unknown/X3'
TRACK2_X4_TRAIN = DATASET_DIR + '/DIV2K_train_LR_unknown/X4'
TRACK1_X2_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_bicubic/X2'
TRACK1_X3_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_bicubic/X3'
TRACK1_X4_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_bicubic/X4'
TRACK2_X2_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_unknown/X2'
TRACK2_X3_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_unknown/X3'
TRACK2_X4_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_unknown/X4'

TRAIN_HR = DATASET_DIR + '/DIV2K_train_HR'
VALIDATION_HR = DATASET_DIR + '/DIV2K_valid_HR'

# PROCESSED IMAGES PATHS
PROCESSED_DIR = './processed'
PROCESSED_TRAIN_DIR = PROCESSED_DIR + '/train'
PROCESSED_VALID_DIR = PROCESSED_DIR + '/valid'

PROCESSED_TRAIN_HR_DIR = PROCESSED_TRAIN_DIR + '/HR'
PROCESSED_TRAIN_X2_DIR = PROCESSED_TRAIN_DIR + '/X2'
PROCESSED_TRAIN_X4_DIR = PROCESSED_TRAIN_DIR + '/X4'

PROCESSED_VALID_HR_DIR = PROCESSED_VALID_DIR + '/HR'
PROCESSED_VALID_X2_DIR = PROCESSED_VALID_DIR + '/X2'
PROCESSED_VALID_X4_DIR = PROCESSED_VALID_DIR + '/X4'

PROCESSED_HR_SHAPE = (512, 512)

# RESULTS PATHS
RESULTS_DIR = './results'
TRACK1_RESULTS_DIR = RESULTS_DIR + '/track1'
TRACK2_RESULTS_DIR = RESULTS_DIR + '/track2'
TRACK1_BICUBIC_RESULTS_DIR = TRACK1_RESULTS_DIR + '/bicubic'
TRACK2_BICUBIC_RESULTS_DIR = TRACK2_RESULTS_DIR + '/bicubic'
TRACK1_ESRGAN_RESULTS_DIR = TRACK1_RESULTS_DIR + '/esrgan'
TRACK2_ESRGAN_RESULTS_DIR = TRACK2_RESULTS_DIR + '/esrgan'

TRACK1_X2_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X2'
TRACK1_X3_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X3'
TRACK1_X4_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X4'

TRACK2_X2_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X2'
TRACK2_X3_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X3'
TRACK2_X4_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X4'

TRACK1_X4_ESRGAN_RESULTS_DIR = TRACK1_ESRGAN_RESULTS_DIR + '/X4'
TRACK2_X4_ESRGAN_RESULTS_DIR = TRACK2_ESRGAN_RESULTS_DIR + '/X4'

# CONSTANTS
SCALE_X2 = 2
SCALE_X3 = 3
SCALE_X4 = 4

# MODELS
MODELS_DIR = './models'
MODEL_RRDB_ESRGAN_X4 = MODELS_DIR + '/RRDB_ESRGAN_X4.pth'

# TARGETS
DATA_PROCESSING_TARGETS = [
    {   
        'name': 'Training Data',
        'raw_dir': TRAIN_HR, 
        'HR_dir': PROCESSED_TRAIN_HR_DIR, 
        'X2_dir': PROCESSED_TRAIN_X2_DIR, 
        'X4_dir': PROCESSED_TRAIN_X4_DIR, 
    }, 
    {      
        'name': 'Validation Data',
        'raw_dir': VALIDATION_HR, 
        'HR_dir': PROCESSED_VALID_HR_DIR, 
        'X2_dir': PROCESSED_VALID_X2_DIR, 
        'X4_dir': PROCESSED_VALID_X4_DIR, 
    }
]

TRACK1_BICUBIC_TARGETS = [
    {
        'name': 'Track 1 - Bicubic Interpolation (x2)',
        'scale': SCALE_X2,
        'test_dir': TRACK1_X2_VALIDATION,
        'results_dir': TRACK1_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 - Bicubic Interpolation (x3)',
        'scale': SCALE_X3,
        'test_dir': TRACK1_X3_VALIDATION,
        'results_dir': TRACK1_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 - Bicubic Interpolation (x4)',
        'scale': SCALE_X4,
        'test_dir': TRACK1_X4_VALIDATION,
        'results_dir': TRACK1_X4_BICUBIC_RESULTS_DIR,
    }
]

TRACK2_BICUBIC_TARGETS = [
    {
        'name': 'Track 2 - Bicubic Interpolation (x2)',
        'scale': SCALE_X2,
        'test_dir': TRACK2_X2_VALIDATION,
        'results_dir': TRACK2_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 - Bicubic Interpolation (x3)',
        'scale': SCALE_X3,
        'test_dir': TRACK2_X3_VALIDATION,
        'results_dir': TRACK2_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 - Bicubic Interpolation (x4)',
        'scale': SCALE_X4,
        'test_dir': TRACK2_X4_VALIDATION,
        'results_dir': TRACK2_X4_BICUBIC_RESULTS_DIR,
    }
]

TRACK1_ESRGANX4_TARGETS = [
    {
        'name': 'Track 1 - ESRGAN (x4)', 
        'scale': SCALE_X4,
        'test_dir': TRACK1_X4_VALIDATION,
        'results_dir': TRACK1_X4_ESRGAN_RESULTS_DIR
    }
]