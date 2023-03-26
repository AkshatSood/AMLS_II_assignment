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
PROCESSED_TRACK1 = PROCESSED_DIR + '/track1'
PROCESSED_TRACK2 = PROCESSED_DIR + '/track2'

PROCESSED_TRACK1_TRAIN_DIR = PROCESSED_TRACK1 + '/train'
PROCESSED_TRACK1_VALID_DIR = PROCESSED_TRACK1 + '/valid'

TRACK1_PROCESSED_TRAIN_HR_DIR = PROCESSED_TRACK1_TRAIN_DIR + '/HR'
TRACK1_PROCESSED_TRAIN_X2_DIR = PROCESSED_TRACK1_TRAIN_DIR + '/X2'
TRACK1_PROCESSED_TRAIN_X4_DIR = PROCESSED_TRACK1_TRAIN_DIR + '/X4'

TRACK1_PROCESSED_VALID_HR_DIR = PROCESSED_TRACK1_VALID_DIR + '/HR'
TRACK1_PROCESSED_VALID_X2_DIR = PROCESSED_TRACK1_VALID_DIR + '/X2'
TRACK1_PROCESSED_VALID_X4_DIR = PROCESSED_TRACK1_VALID_DIR + '/X4'

PROCESSED_TRACK2_TRAIN_DIR = PROCESSED_TRACK2 + '/train'
PROCESSED_TRACK2_VALID_DIR = PROCESSED_TRACK2 + '/valid'

TRACK2_PROCESSED_TRAIN_HR_DIR = PROCESSED_TRACK2_TRAIN_DIR + '/HR'
TRACK2_PROCESSED_TRAIN_X2_DIR = PROCESSED_TRACK2_TRAIN_DIR + '/X2'
TRACK2_PROCESSED_TRAIN_X4_DIR = PROCESSED_TRACK2_TRAIN_DIR + '/X4'

TRACK2_PROCESSED_VALID_HR_DIR = PROCESSED_TRACK2_VALID_DIR + '/HR'
TRACK2_PROCESSED_VALID_X2_DIR = PROCESSED_TRACK2_VALID_DIR + '/X2'
TRACK2_PROCESSED_VALID_X4_DIR = PROCESSED_TRACK2_VALID_DIR + '/X4'

PROCESSED_HR_SHAPE = (512, 512)

# RESULTS PATHS
RESULTS_DIR = './results'
TRACK1_RESULTS_DIR = RESULTS_DIR + '/track1'
TRACK2_RESULTS_DIR = RESULTS_DIR + '/track2'
TRACK1_CROPPED_RESULTS_DIR = RESULTS_DIR + '/track1_cropped'
TRACK2_CROPPED_RESULTS_DIR = RESULTS_DIR + '/track2_cropped'

TRACK1_BICUBIC_RESULTS_DIR = TRACK1_RESULTS_DIR + '/bicubic'
TRACK2_BICUBIC_RESULTS_DIR = TRACK2_RESULTS_DIR + '/bicubic'
TRACK1_CROPPED_BICUBIC_RESULTS_DIR = TRACK1_CROPPED_RESULTS_DIR + '/bicubic'
TRACK2_CROPPED_BICUBIC_RESULTS_DIR = TRACK2_CROPPED_RESULTS_DIR + '/bicubic'

TRACK1_REAL_ESRGAN_RESULTS_DIR = TRACK1_RESULTS_DIR + '/real_esrgan'
TRACK2_REAL_ESRGAN_RESULTS_DIR = TRACK2_RESULTS_DIR + '/real_esrgan'
TRACK1_CROPPED_REAL_ESRGAN_RESULTS_DIR = TRACK1_CROPPED_RESULTS_DIR + '/real_esrgan'
TRACK2_CROPPED_REAL_ESRGAN_RESULTS_DIR = TRACK2_CROPPED_RESULTS_DIR + '/real_esrgan'

TRACK1_X2_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X2'
TRACK1_X3_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X3'
TRACK1_X4_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X4'

TRACK2_X2_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X2'
TRACK2_X3_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X3'
TRACK2_X4_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X4'

TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR = TRACK1_CROPPED_BICUBIC_RESULTS_DIR + '/X2'
TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR = TRACK2_CROPPED_BICUBIC_RESULTS_DIR + '/X2'
TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR = TRACK1_CROPPED_BICUBIC_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_BICUBIC_RESULTS_DIR = TRACK2_CROPPED_BICUBIC_RESULTS_DIR + '/X4'

TRACK1_X4_ESRGAN_RESULTS_DIR = TRACK1_REAL_ESRGAN_RESULTS_DIR + '/X4'
TRACK2_X4_ESRGAN_RESULTS_DIR = TRACK2_REAL_ESRGAN_RESULTS_DIR + '/X4'
TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR = TRACK1_CROPPED_REAL_ESRGAN_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_ESRGAN_RESULTS_DIR = TRACK2_CROPPED_REAL_ESRGAN_RESULTS_DIR + '/X4'

# CONSTANTS
SCALE_X1 = 1
SCALE_X2 = 2
SCALE_X3 = 3
SCALE_X4 = 4
PROGRESS_NUM = 10

# MODELS
MODELS_DIR = './models'
MODEL_RRDB_ESRGAN_X4 = MODELS_DIR + '/RRDB_ESRGAN_X4.pth'

# TARGETS
DATA_PROCESSING_TARGETS = [
    {   
        'name': 'Track 1 - Training Data (HR)', 
        'raw_dir': TRAIN_HR, 
        'output_dir': TRACK1_PROCESSED_TRAIN_HR_DIR,
        'scale': SCALE_X1
    }, 
    {   
        'name': 'Track 1 - Training Data (X2)', 
        'raw_dir': TRACK1_X2_TRAIN, 
        'output_dir': TRACK1_PROCESSED_TRAIN_X2_DIR,
        'scale': SCALE_X2
    },
    {   
        'name': 'Track 1 - Training Data (X4)', 
        'raw_dir': TRACK1_X4_TRAIN, 
        'output_dir': TRACK1_PROCESSED_TRAIN_X4_DIR,
        'scale': SCALE_X4
    },
    {   
        'name': 'Track 1 - Validation Data (HR)', 
        'raw_dir': VALIDATION_HR, 
        'output_dir': TRACK1_PROCESSED_VALID_HR_DIR,
        'scale': SCALE_X1
    }, 
    {   
        'name': 'Track 1 - Validation Data (X2)', 
        'raw_dir': TRACK1_X2_VALIDATION, 
        'output_dir': TRACK1_PROCESSED_VALID_X2_DIR,
        'scale': SCALE_X2
    },
    {   
        'name': 'Track 1 - Validation Data (X4)', 
        'raw_dir': TRACK1_X4_VALIDATION, 
        'output_dir': TRACK1_PROCESSED_VALID_X4_DIR,
        'scale': SCALE_X4
    },
    {   
        'name': 'Track 2 - Training Data (HR)', 
        'raw_dir': TRAIN_HR, 
        'output_dir': TRACK2_PROCESSED_TRAIN_HR_DIR,
        'scale': SCALE_X1
    }, 
    {   
        'name': 'Track 2 - Training Data (X2)', 
        'raw_dir': TRACK1_X2_TRAIN, 
        'output_dir': TRACK2_PROCESSED_TRAIN_X2_DIR,
        'scale': SCALE_X2
    },
    {   
        'name': 'Track 2 - Training Data (X4)', 
        'raw_dir': TRACK2_X4_TRAIN, 
        'output_dir': TRACK2_PROCESSED_TRAIN_X4_DIR,
        'scale': SCALE_X4
    },
    {   
        'name': 'Track 2 - Validation Data (HR)', 
        'raw_dir': VALIDATION_HR, 
        'output_dir': TRACK2_PROCESSED_VALID_HR_DIR,
        'scale': SCALE_X1
    }, 
    {   
        'name': 'Track 2 - Validation Data (X2)', 
        'raw_dir': TRACK2_X2_VALIDATION, 
        'output_dir': TRACK2_PROCESSED_VALID_X2_DIR,
        'scale': SCALE_X2
    },
    {   
        'name': 'Track 2 - Validation Data (X4)', 
        'raw_dir': TRACK2_X4_VALIDATION, 
        'output_dir': TRACK2_PROCESSED_VALID_X4_DIR,
        'scale': SCALE_X4
    },
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

CROPPED_BICUBIC_TARGETS = []
# CROPPED_BICUBIC_TARGETS = [
#     {
#         'name': 'Custom - Bicubic Interpolation (x2)',
#         'scale': SCALE_X2,
#         'test_dir': PROCESSED_VALID_X2_DIR,
#         'results_dir': CROPPED_X2_BICUBIC_RESULTS_DIR,
#     },
#     {
#         'name': 'Custom - Bicubic Interpolation (x4)',
#         'scale': SCALE_X4,
#         'test_dir': PROCESSED_VALID_X4_DIR,
#         'results_dir': CROPPED_X4_BICUBIC_RESULTS_DIR,
#     }
# ]

TRACK1_ESRGANX4_TARGETS = [
    # {
    #     'name': 'Track 1 - ESRGAN (x4)', 
    #     'scale': SCALE_X4,
    #     'test_dir': TRACK1_X4_VALIDATION,
    #     'results_dir': TRACK1_X4_ESRGAN_RESULTS_DIR
    # },
    {
        'name': 'Track 1 (Cropped) - ESRGAN (x4)', 
        'scale': SCALE_X4,
        'test_dir': TRACK1_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR
    }
]

TRACK2_ESRGANX4_TARGETS = [
    {
        'name': 'Track 2 - ESRGAN (x4)', 
        'scale': SCALE_X4,
        'test_dir': TRACK2_X4_VALIDATION,
        'results_dir': TRACK2_X4_ESRGAN_RESULTS_DIR
    }
]

CROPPED_ESRGANX4_TARGETS = []
# CROPPED_ESRGANX4_TARGETS = [
#     {
#         'name': 'Custom - ESRGAN (x4)',
#         'scale': SCALE_X4,
#         'test_dir': PROCESSED_VALID_X4_DIR,
#         'results_dir': CROPPED_X4_ESRGAN_RESULTS_DIR
#     }
# ]
