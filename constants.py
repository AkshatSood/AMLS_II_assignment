"""Constants Module"""

# DATASET PATHS
DATASET_DIR = './dataset'
TRAIN_HR = DATASET_DIR + '/DIV2K_train_HR'
TRACK1_X2_TRAIN = DATASET_DIR + '/DIV2K_train_LR_bicubic/X2'
TRACK1_X3_TRAIN = DATASET_DIR + '/DIV2K_train_LR_bicubic/X3'
TRACK1_X4_TRAIN = DATASET_DIR + '/DIV2K_train_LR_bicubic/X4'
TRACK2_X2_TRAIN = DATASET_DIR + '/DIV2K_train_LR_unknown/X2'
TRACK2_X3_TRAIN = DATASET_DIR + '/DIV2K_train_LR_unknown/X3'
TRACK2_X4_TRAIN = DATASET_DIR + '/DIV2K_train_LR_unknown/X4'
VALIDATION_HR = DATASET_DIR + '/DIV2K_valid_HR'
TRACK1_X2_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_bicubic/X2'
TRACK1_X3_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_bicubic/X3'
TRACK1_X4_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_bicubic/X4'
TRACK2_X2_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_unknown/X2'
TRACK2_X3_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_unknown/X3'
TRACK2_X4_VALIDATION = DATASET_DIR + '/DIV2K_valid_LR_unknown/X4'

# H5 FILE PATHS
H5_TRACK1_X2_TRAIN = DATASET_DIR + '/DIV2K_track1_X2_train.h5'
H5_TRACK1_X3_TRAIN = DATASET_DIR + '/DIV2K_track1_X3_train.h5'
H5_TRACK1_X4_TRAIN = DATASET_DIR + '/DIV2K_track1_X4_train.h5'
H5_TRACK2_X2_TRAIN = DATASET_DIR + '/DIV2K_track2_X2_train.h5'
H5_TRACK2_X3_TRAIN = DATASET_DIR + '/DIV2K_track2_X3_train.h5'
H5_TRACK2_X4_TRAIN = DATASET_DIR + '/DIV2K_track2_X4_train.h5'
H5_TRACK1_X2_VALID = DATASET_DIR + '/DIV2K_track1_X2_valid.h5'
H5_TRACK1_X3_VALID = DATASET_DIR + '/DIV2K_track1_X3_valid.h5'
H5_TRACK1_X4_VALID = DATASET_DIR + '/DIV2K_track1_X4_valid.h5'
H5_TRACK2_X2_VALID = DATASET_DIR + '/DIV2K_track2_X2_valid.h5'
H5_TRACK2_X3_VALID = DATASET_DIR + '/DIV2K_track2_X3_valid.h5'
H5_TRACK2_X4_VALID = DATASET_DIR + '/DIV2K_track2_X4_valid.h5'
# H5_TRACK1_X2_TRAIN = 'D:/AMLS2' + '/DIV2K_track1_X2_train.h5'
# H5_TRACK1_X3_TRAIN = 'D:/AMLS2' + '/DIV2K_track1_X3_train.h5'
# H5_TRACK1_X4_TRAIN = 'D:/AMLS2' + '/DIV2K_track1_X4_train.h5'
# H5_TRACK2_X2_TRAIN = 'E:/AMLS2' + '/DIV2K_track2_X2_train.h5'
# H5_TRACK2_X3_TRAIN = 'E:/AMLS2' + '/DIV2K_track2_X3_train.h5'
# H5_TRACK2_X4_TRAIN = 'E:/AMLS2' + '/DIV2K_track2_X4_train.h5'
# H5_TRACK1_X2_VALID = 'D:/AMLS2' + '/DIV2K_track1_X2_valid.h5'
# H5_TRACK1_X3_VALID = 'D:/AMLS2' + '/DIV2K_track1_X3_valid.h5'
# H5_TRACK1_X4_VALID = 'D:/AMLS2' + '/DIV2K_track1_X4_valid.h5'
# H5_TRACK2_X2_VALID = 'E:/AMLS2' + '/DIV2K_track2_X2_valid.h5'
# H5_TRACK2_X3_VALID = 'E:/AMLS2' + '/DIV2K_track2_X3_valid.h5'
# H5_TRACK2_X4_VALID = 'E:/AMLS2' + '/DIV2K_track2_X4_valid.h5'

# PROCESSED IMAGES PATHS
PROCESSED_DIR = './processed'
PROCESSED_TRACK1 = PROCESSED_DIR + '/track1'
PROCESSED_TRACK2 = PROCESSED_DIR + '/track2'
PROCESSED_TRACK1_TRAIN_DIR = PROCESSED_TRACK1 + '/train'
PROCESSED_TRACK1_VALID_DIR = PROCESSED_TRACK1 + '/valid'
PROCESSED_TRACK2_TRAIN_DIR = PROCESSED_TRACK2 + '/train'
PROCESSED_TRACK2_VALID_DIR = PROCESSED_TRACK2 + '/valid'

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

TRACK1_ESRGAN_RESULTS_DIR = TRACK1_RESULTS_DIR + '/esrgan'
TRACK2_ESRGAN_RESULTS_DIR = TRACK2_RESULTS_DIR + '/esrgan'
TRACK1_CROPPED_ESRGAN_RESULTS_DIR = TRACK1_CROPPED_RESULTS_DIR + '/esrgan_esrgan'
TRACK2_CROPPED_ESRGAN_RESULTS_DIR = TRACK2_CROPPED_RESULTS_DIR + '/esrgan_esrgan'
TRACK1_CROPPED_ESRGAN_PSNR_RESULTS_DIR = TRACK1_CROPPED_RESULTS_DIR + '/esrgan_psnr'
TRACK2_CROPPED_ESRGAN_PSNR_RESULTS_DIR = TRACK2_CROPPED_RESULTS_DIR + '/esrgan_psnr'

TRACK1_CROPPED_SRCNN_RESULTS_DIR = TRACK1_CROPPED_RESULTS_DIR + '/srcnn'
TRACK2_CROPPED_SRCNN_RESULTS_DIR = TRACK2_CROPPED_RESULTS_DIR + '/srcnn'

TRACK1_CROPPED_FSRCNN_RESULTS_DIR = TRACK1_CROPPED_RESULTS_DIR + '/fsrcnn'
TRACK2_CROPPED_FSRCNN_RESULTS_DIR = TRACK2_CROPPED_RESULTS_DIR + '/fsrcnn'

TRACK1_X2_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X2'
TRACK1_X3_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X3'
TRACK1_X4_BICUBIC_RESULTS_DIR = TRACK1_BICUBIC_RESULTS_DIR + '/X4'

TRACK2_X2_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X2'
TRACK2_X3_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X3'
TRACK2_X4_BICUBIC_RESULTS_DIR = TRACK2_BICUBIC_RESULTS_DIR + '/X4'

TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR = TRACK1_CROPPED_BICUBIC_RESULTS_DIR + '/X2'
TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR = TRACK2_CROPPED_BICUBIC_RESULTS_DIR + '/X2'
TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR = TRACK1_CROPPED_BICUBIC_RESULTS_DIR + '/X3'
TRACK2_CROPPED_X3_BICUBIC_RESULTS_DIR = TRACK2_CROPPED_BICUBIC_RESULTS_DIR + '/X3'
TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR = TRACK1_CROPPED_BICUBIC_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_BICUBIC_RESULTS_DIR = TRACK2_CROPPED_BICUBIC_RESULTS_DIR + '/X4'

TRACK1_X4_ESRGAN_RESULTS_DIR = TRACK1_ESRGAN_RESULTS_DIR + '/X4'
TRACK2_X4_ESRGAN_RESULTS_DIR = TRACK2_ESRGAN_RESULTS_DIR + '/X4'
TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR = TRACK1_CROPPED_ESRGAN_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_ESRGAN_RESULTS_DIR = TRACK2_CROPPED_ESRGAN_RESULTS_DIR + '/X4'
TRACK1_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR = TRACK1_CROPPED_ESRGAN_PSNR_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR = TRACK2_CROPPED_ESRGAN_PSNR_RESULTS_DIR + '/X4'

TRACK1_CROPPED_X2_SRCNN_RESULTS_DIR = TRACK1_CROPPED_SRCNN_RESULTS_DIR + '/X2'
TRACK2_CROPPED_X2_SRCNN_RESULTS_DIR = TRACK2_CROPPED_SRCNN_RESULTS_DIR + '/X2'
TRACK1_CROPPED_X3_SRCNN_RESULTS_DIR = TRACK1_CROPPED_SRCNN_RESULTS_DIR + '/X3'
TRACK2_CROPPED_X3_SRCNN_RESULTS_DIR = TRACK2_CROPPED_SRCNN_RESULTS_DIR + '/X3'
TRACK1_CROPPED_X4_SRCNN_RESULTS_DIR = TRACK1_CROPPED_SRCNN_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_SRCNN_RESULTS_DIR = TRACK2_CROPPED_SRCNN_RESULTS_DIR + '/X4'

TRACK1_CROPPED_X2_FSRCNN_RESULTS_DIR = TRACK1_CROPPED_FSRCNN_RESULTS_DIR + '/X2'
TRACK2_CROPPED_X2_FSRCNN_RESULTS_DIR = TRACK2_CROPPED_FSRCNN_RESULTS_DIR + '/X2'
TRACK1_CROPPED_X3_FSRCNN_RESULTS_DIR = TRACK1_CROPPED_FSRCNN_RESULTS_DIR + '/X3'
TRACK2_CROPPED_X3_FSRCNN_RESULTS_DIR = TRACK2_CROPPED_FSRCNN_RESULTS_DIR + '/X3'
TRACK1_CROPPED_X4_FSRCNN_RESULTS_DIR = TRACK1_CROPPED_FSRCNN_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_FSRCNN_RESULTS_DIR = TRACK2_CROPPED_FSRCNN_RESULTS_DIR + '/X4'

EVALUATION_DIR = './evaluation'

# CONSTANTS
SCALE_X1 = 1
SCALE_X2 = 2
SCALE_X3 = 3
SCALE_X4 = 4
PROGRESS_NUM = 10
PROCESSED_HR_SHAPE = (648, 648)
EVALUATION_IMG_SHAPE = (630, 630)
SRCNN_X2_BORDER = 7
SRCNN_X3_BORDER = 7
SRCNN_X4_BORDER = 7

# MODELS
MODELS_DIR = './models'
MODEL_RRDB_ESRGAN_X4 = MODELS_DIR + '/RRDB_ESRGAN_X4.pth'
MODEL_RRDB_PSNR_X4 = MODELS_DIR + '/RRDB_PSNR_x4.pth'
MODEL_SRCNN_WEIGHTS = MODELS_DIR + '/3051crop_weight_200.h5'
MODEL_FSRCNN_X2_WEIGHTS = MODELS_DIR + '/fsrcnn_x2.pth'
MODEL_FSRCNN_X3_WEIGHTS = MODELS_DIR + '/fsrcnn_x3.pth'
MODEL_FSRCNN_X4_WEIGHTS = MODELS_DIR + '/fsrcnn_x4.pth'
WEIGHTS_DIR_FSRCNN_TRACK1 = './models/fsrcnn/track1/'
WEIGHTS_DIR_FSRCNN_TRACK2 = './models/fsrcnn/track2/'

# TARGETS
DATASET_DOWNLOAD_TARGETS = [
    {
        'name': 'DIV2K Train HR',
        'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'target_dir': TRAIN_HR,
    },
    {
        'name': 'DIV2K Valid HR',
        'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        'target_dir': VALIDATION_HR,
    },
    # {
    #     'name': 'DIV2K Train LR Track 1 (X2)',
    #     'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip',
    #     'target_dir': TRACK1_X2_TRAIN,
    # },
    # {
    #     'name': 'DIV2K Train LR Track 1 (X3)',
    #     'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip',
    #     'target_dir': TRACK1_X3_TRAIN,
    # },
    {
        'name': 'DIV2K Train LR Track 1 (X4)',
        'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip',
        'target_dir': TRACK1_X4_TRAIN,
    },
    # {
    #     'name': 'DIV2K Validation LR Track 1 (X2)',
    #     'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip',
    #     'target_dir': TRACK1_X2_VALIDATION,
    # },
    # {
    #     'name': 'DIV2K Validation LR Track 1 (X3)',
    #     'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip',
    #     'target_dir': TRACK1_X3_VALIDATION,
    # },
    {
        'name': 'DIV2K Validation LR Track 1 (X4)',
        'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip',
        'target_dir': TRACK1_X4_VALIDATION,
    },
    # {
    #     'name': 'DIV2K Train LR Track 2 (X2)',
    #     'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X2.zip',
    #     'target_dir': TRACK2_X2_TRAIN,
    # },
    # {
    #     'name': 'DIV2K Train LR Track 2 (X3)',
    #     'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X3.zip',
    #     'target_dir': TRACK2_X3_TRAIN,
    # },
    {
        'name': 'DIV2K Train LR Track 2 (X4)',
        'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip',
        'target_dir': TRACK2_X4_TRAIN,
    },
    # {
    #     'name': 'DIV2K Validation LR Track 2 (X2)',
    #     'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X2.zip',
    #     'target_dir': TRACK2_X2_VALIDATION,
    # },
    # {
    #     'name': 'DIV2K Validation LR Track 2 (X3)',
    #     'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X3.zip',
    #     'target_dir': TRACK2_X3_VALIDATION,
    # },
    {
        'name': 'DIV2K Validation LR Track 2 (X4)',
        'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip',
        'target_dir': TRACK2_X4_VALIDATION,
    }
]

DATA_PROCESSING_CROP_TARGETS = [
    {
        'name': 'Track 1 - Training Data (HR)',
        'src_dir': TRAIN_HR,
        'output_dir': PROCESSED_TRACK1_TRAIN_DIR + '/HR',
        'scale': SCALE_X1,
    },
    # {
    #     'name': 'Track 1 - Training Data (X2)',
    #     'src_dir': TRACK1_X2_TRAIN,
    #     'output_dir': PROCESSED_TRACK1_TRAIN_DIR + '/X2',
    #     'scale': SCALE_X2,        
    # },
    # {
    #     'name': 'Track 1 - Training Data (X3)',
    #     'src_dir': TRACK1_X3_TRAIN,
    #     'output_dir': PROCESSED_TRACK1_TRAIN_DIR + '/X3',
    #     'scale': SCALE_X3,
    # },
    {
        'name': 'Track 1 - Training Data (X4)',
        'src_dir': TRACK1_X4_TRAIN,
        'output_dir': PROCESSED_TRACK1_TRAIN_DIR + '/X4',
        'scale': SCALE_X4,
    },
    {
        'name': 'Track 1 - Validation Data (HR)',
        'src_dir': VALIDATION_HR,
        'output_dir': PROCESSED_TRACK1_VALID_DIR + '/HR',
        'scale': SCALE_X1,
    },
    # {
    #     'name': 'Track 1 - Validation Data (X2)',
    #     'src_dir': TRACK1_X2_VALIDATION,
    #     'output_dir': PROCESSED_TRACK1_VALID_DIR + '/X2',
    #     'scale': SCALE_X2,
    # },
    # {
    #     'name': 'Track 1 - Validation Data (X3)',
    #     'src_dir': TRACK1_X3_VALIDATION,
    #     'output_dir': PROCESSED_TRACK1_VALID_DIR + '/X3',
    #     'scale': SCALE_X3,
    # },
    {
        'name': 'Track 1 - Validation Data (X4)',
        'src_dir': TRACK1_X4_VALIDATION,
        'output_dir': PROCESSED_TRACK1_VALID_DIR + '/X4',
        'scale': SCALE_X4,
    },
    {
        'name': 'Track 2 - Training Data (HR)',
        'src_dir': TRAIN_HR,
        'output_dir': PROCESSED_TRACK2_TRAIN_DIR + '/HR',
        'scale': SCALE_X1,
    },
    # {
    #     'name': 'Track 2 - Training Data (X2)',
    #     'src_dir': TRACK2_X2_TRAIN,
    #     'output_dir': PROCESSED_TRACK2_TRAIN_DIR + '/X2',
    #     'scale': SCALE_X2,
    # },
    # {
    #     'name': 'Track 2 - Training Data (X3)',
    #     'src_dir': TRACK2_X3_TRAIN,
    #     'output_dir': PROCESSED_TRACK2_TRAIN_DIR + '/X3',
    #     'scale': SCALE_X3,
    # },
    {
        'name': 'Track 2 - Training Data (X4)',
        'src_dir': TRACK2_X4_TRAIN,
        'output_dir': PROCESSED_TRACK2_TRAIN_DIR + '/X4',
        'scale': SCALE_X4,
    },
    {
        'name': 'Track 2 - Validation Data (HR)',
        'src_dir': VALIDATION_HR,
        'output_dir': PROCESSED_TRACK2_VALID_DIR + '/HR',
        'scale': SCALE_X1,
    },
    # {
    #     'name': 'Track 2 - Validation Data (X2)',
    #     'src_dir': TRACK2_X2_VALIDATION,
    #     'output_dir': PROCESSED_TRACK2_VALID_DIR + '/X2',
    #     'scale': SCALE_X2,
    # },
    # {
    #     'name': 'Track 2 - Validation Data (X3)',
    #     'src_dir': TRACK2_X3_VALIDATION,
    #     'output_dir': PROCESSED_TRACK2_VALID_DIR + '/X3',
    #     'scale': SCALE_X3,
    # },
    {
        'name': 'Track 2 - Validation Data (X4)',
        'src_dir': TRACK2_X4_VALIDATION,
        'output_dir': PROCESSED_TRACK2_VALID_DIR + '/X4',
        'scale': SCALE_X4,
    }
]

DATA_PROCESSING_H5_TARGETS = [
    # {
    #     'name': 'Track 1 - Training Data (X2)',
    #     'h5': H5_TRACK1_X2_TRAIN,
    #     'lr_dir': TRACK1_X2_TRAIN,
    #     'hr_dir': TRAIN_HR, 
    #     'scale': SCALE_X2,
    #     'training': True,
    # }, 
    # {
    #     'name': 'Track 1 - Training Data (X3)',
    #     'h5': H5_TRACK1_X3_TRAIN,
    #     'lr_dir': TRACK1_X3_TRAIN,
    #     'hr_dir': TRAIN_HR, 
    #     'scale': SCALE_X3,
    #     'training': True,
    # },
    {
        'name': 'Track 1 - Training Data (X4)',
        'h5': H5_TRACK1_X4_TRAIN,
        'lr_dir': PROCESSED_TRACK1_TRAIN_DIR + '/X4',
        'hr_dir': PROCESSED_TRACK1_TRAIN_DIR + '/HR', 
        'scale': SCALE_X4,
        'training': True,
    },
    # {
    #     'name': 'Track 1 - Validation Data (X2)',
    #     'h5': H5_TRACK1_X2_VALID,
    #     'lr_dir': TRACK1_X2_VALIDATION,
    #     'hr_dir': VALIDATION_HR, 
    #     'scale': SCALE_X2,
    #     'training': False
    # }, 
    # {
    #     'name': 'Track 1 - Validation Data (X3)',
    #     'h5': H5_TRACK1_X3_VALID,
    #     'lr_dir': TRACK1_X3_VALIDATION,
    #     'hr_dir': VALIDATION_HR, 
    #     'scale': SCALE_X3,
    #     'training': False
    # },
    {
        'name': 'Track 1 - Validation Data (X4)',
        'h5': H5_TRACK1_X4_VALID,
        'lr_dir': PROCESSED_TRACK1_VALID_DIR + '/X4',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'scale': SCALE_X4,
        'training': False
    },
    # {
    #     'name': 'Track 2 - Training Data (X2)',
    #     'h5': H5_TRACK2_X2_TRAIN,
    #     'lr_dir': TRACK2_X2_TRAIN,
    #     'hr_dir': TRAIN_HR, 
    #     'scale': SCALE_X2,
    #     'training': True
    # }, 
    # {
    #     'name': 'Track 2 - Training Data (X3)',
    #     'h5': H5_TRACK2_X3_TRAIN,
    #     'lr_dir': TRACK2_X3_TRAIN,
    #     'hr_dir': TRAIN_HR, 
    #     'scale': SCALE_X3,
    #     'training': True
    # },
    {
        'name': 'Track 2 - Training Data (X4)',
        'h5': H5_TRACK2_X4_TRAIN,
        'lr_dir': PROCESSED_TRACK2_TRAIN_DIR + '/X4',
        'hr_dir': PROCESSED_TRACK2_TRAIN_DIR + '/HR', 
        'scale': SCALE_X4,
        'training': True
    },
    # {
    #     'name': 'Track 2 - Validation Data (X2)',
    #     'h5': H5_TRACK2_X2_VALID,
    #     'lr_dir': TRACK2_X2_VALIDATION,
    #     'hr_dir': VALIDATION_HR, 
    #     'scale': SCALE_X2,
    #     'training': False
    # }, 
    # {
    #     'name': 'Track 2 - Validation Data (X3)',
    #     'h5': H5_TRACK2_X3_VALID,
    #     'lr_dir': TRACK2_X3_VALIDATION,
    #     'hr_dir': VALIDATION_HR, 
    #     'scale': SCALE_X3,
    #     'training': False
    # },
    {
        'name': 'Track 2 - Validation Data (X4)',
        'h5': H5_TRACK2_X4_VALID,
        'lr_dir': PROCESSED_TRACK2_VALID_DIR + '/X4',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'scale': SCALE_X4,
        'training': False
    },
]

TRAIN_FSRCNN_TARGETS = [
    # {
    #     'name': 'Track 1 - Training Data (X2)',
    #     'train_h5': H5_TRACK1_X2_TRAIN,
    #     'valid_h5': H5_TRACK1_X2_VALID,
    #     'weights_dir': WEIGHTS_DIR_FSRCNN_TRACK1, 
    #     'prefix': 'track1_x2_',
    #     'scale': SCALE_X2
    # },
    # {
    #     'name': 'Track 1 - Training Data (X3)',
    #     'train_h5': H5_TRACK1_X3_TRAIN,
    #     'valid_h5': H5_TRACK1_X3_VALID,
    #     'weights_dir': WEIGHTS_DIR_FSRCNN_TRACK1, 
    #     'prefix': 'track1_x3_',
    #     'scale': SCALE_X3
    # },
    {
        'name': 'Track 1 - Training Data (X4)',
        'train_h5': H5_TRACK1_X4_TRAIN,
        'valid_h5': H5_TRACK1_X4_VALID,
        'weights_dir': WEIGHTS_DIR_FSRCNN_TRACK1, 
        'prefix': 'track1_x4_',
        'scale': SCALE_X4
    },
    # {
    #     'name': 'Track 2 - Training Data (X2)',
    #     'train_h5': H5_TRACK2_X2_TRAIN,
    #     'valid_h5': H5_TRACK2_X2_VALID,
    #     'weights_dir': WEIGHTS_DIR_FSRCNN_TRACK2, 
    #     'prefix': 'track2_x2_',
    #     'scale': SCALE_X2
    # },
    # {
    #     'name': 'Track 2 - Training Data (X3)',
    #     'train_h5': H5_TRACK2_X3_TRAIN,
    #     'valid_h5': H5_TRACK2_X3_VALID,
    #     'weights_dir': WEIGHTS_DIR_FSRCNN_TRACK2, 
    #     'prefix': 'track2_x3_',
    #     'scale': SCALE_X3
    # },
    {
        'name': 'Track 2 - Training Data (X4)',
        'train_h5': H5_TRACK2_X4_TRAIN,
        'valid_h5': H5_TRACK2_X4_VALID,
        'weights_dir': WEIGHTS_DIR_FSRCNN_TRACK2, 
        'prefix': 'track2_x4_',
        'scale': SCALE_X4
    }
]

TRACK1_BICUBIC_TARGETS = [
    {
        'name': 'Track 1 - Bicubic Interpolation (x2) [Validation]',
        'scale': SCALE_X2,
        'src_dir': TRACK1_X2_VALIDATION,
        'results_dir': TRACK1_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 - Bicubic Interpolation (x3) [Validation]',
        'scale': SCALE_X3,
        'src_dir': TRACK1_X3_VALIDATION,
        'results_dir': TRACK1_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 - Bicubic Interpolation (x4) [Validation]',
        'scale': SCALE_X4,
        'src_dir': TRACK1_X4_VALIDATION,
        'results_dir': TRACK1_X4_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 (Cropped) - Bicubic Interpolation (x2) [Validation]',
        'scale': SCALE_X2,
        'src_dir': PROCESSED_TRACK1_VALID_DIR + '/X2',
        'results_dir': TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 (Cropped) - Bicubic Interpolation (x3) [Validation]',
        'scale': SCALE_X3,
        'src_dir': PROCESSED_TRACK1_VALID_DIR + '/X3',
        'results_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 (Cropped) - Bicubic Interpolation (x4) [Validation]',
        'scale': SCALE_X4,
        'src_dir': PROCESSED_TRACK1_VALID_DIR + '/X4',
        'results_dir': TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR,
    },
]

TRACK2_BICUBIC_TARGETS = [
    {
        'name': 'Track 2 - Bicubic Interpolation (x2) [Validation]',
        'scale': SCALE_X2,
        'src_dir': TRACK2_X2_VALIDATION,
        'results_dir': TRACK2_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 - Bicubic Interpolation (x3) [Validation]',
        'scale': SCALE_X3,
        'src_dir': TRACK2_X3_VALIDATION,
        'results_dir': TRACK2_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 - Bicubic Interpolation (x4) [Validation]',
        'scale': SCALE_X4,
        'src_dir': TRACK2_X4_VALIDATION,
        'results_dir': TRACK2_X4_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 (Cropped) - Bicubic Interpolation (x2) [Validation]',
        'scale': SCALE_X2,
        'src_dir': PROCESSED_TRACK2_VALID_DIR + '/X2',
        'results_dir': TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 (Cropped) - Bicubic Interpolation (x3) [Validation]',
        'scale': SCALE_X3,
        'src_dir': PROCESSED_TRACK2_VALID_DIR + '/X3',
        'results_dir': TRACK2_CROPPED_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 (Cropped) - Bicubic Interpolation (x4) [Validation]',
        'scale': SCALE_X4,
        'src_dir': PROCESSED_TRACK2_VALID_DIR + '/X4',
        'results_dir': TRACK2_CROPPED_X4_BICUBIC_RESULTS_DIR,
    },
]

TRACK1_ESRGANX4_TARGETS = [
    {
        'name': 'Track 1 (Cropped) - ESRGAN (ESRGAN) (X4) [Validation]',
        'scale': SCALE_X4,
        'model': MODEL_RRDB_ESRGAN_X4,
        'src_dir': PROCESSED_TRACK1_VALID_DIR + '/X4',
        'results_dir': TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - ESRGAN (PSNR) (X4) [Validation]',
        'scale': SCALE_X4,
        'model': MODEL_RRDB_PSNR_X4,
        'src_dir': PROCESSED_TRACK1_VALID_DIR + '/X4',
        'results_dir': TRACK1_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR
    }
]

TRACK2_ESRGANX4_TARGETS = [
    {
        'name': 'Track 2 (Cropped) - ESRGAN (ESRGAN) (x4) [Validation]',
        'scale': SCALE_X4,
        'model': MODEL_RRDB_ESRGAN_X4,
        'src_dir': PROCESSED_TRACK2_VALID_DIR + '/X4',
        'results_dir': TRACK2_CROPPED_X4_ESRGAN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - ESRGAN (PSNR) (x4) [Validation]',
        'scale': SCALE_X4,
        'model': MODEL_RRDB_PSNR_X4,
        'src_dir': PROCESSED_TRACK2_VALID_DIR + '/X4',
        'results_dir': TRACK2_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR
    }
]

TRACK1_SRCNN_TARGETS = [
    {
        'name': 'Track 1 (Cropped) - SRCNN (X2) [Validation]',
        'scale': SCALE_X2,
        'border': SRCNN_X2_BORDER,
        'src_dir': TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK1_CROPPED_X2_SRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - SRCNN (X3) [Validation]',
        'scale': SCALE_X3,
        'border': SRCNN_X3_BORDER,
        'src_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK1_CROPPED_X3_SRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - SRCNN (X4) [Validation]',
        'scale': SCALE_X4,
        'border': SRCNN_X4_BORDER,
        'src_dir': TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK1_CROPPED_X4_SRCNN_RESULTS_DIR
    }
]

TRACK2_SRCNN_TARGETS = [
    {
        'name': 'Track 2 (Cropped) - SRCNN (X2) [Validation]',
        'scale': SCALE_X2,
        'border': SRCNN_X2_BORDER,
        'src_dir': TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK2_CROPPED_X2_SRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - SRCNN (X3) [Validation]',
        'scale': SCALE_X3,
        'border': SRCNN_X3_BORDER,
        'src_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK2_CROPPED_X3_SRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - SRCNN (X4) [Validation]',
        'scale': SCALE_X4,
        'border': SRCNN_X4_BORDER,
        'src_dir': TRACK2_CROPPED_X4_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK2_CROPPED_X4_SRCNN_RESULTS_DIR
    }
]

TRACK1_FSRCNN_TARGETS = [
    {
        'name': 'Track 1 (Cropped) - FSRCNN (X2) [Validation]',
        'scale': SCALE_X2,
        'weights_file': MODEL_FSRCNN_X2_WEIGHTS,
        'src_dir': PROCESSED_TRACK1_VALID_DIR + '/X2',
        'results_dir': TRACK1_CROPPED_X2_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - FSRCNN (X3) [Validation]',
        'scale': SCALE_X3,
        'weights_file': MODEL_FSRCNN_X3_WEIGHTS,
        'src_dir': PROCESSED_TRACK1_VALID_DIR + '/X3',
        'results_dir': TRACK1_CROPPED_X3_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - FSRCNN (X4) [Validation]',
        'scale': SCALE_X4,
        'weights_file': MODEL_FSRCNN_X4_WEIGHTS,
        'src_dir': PROCESSED_TRACK1_VALID_DIR + '/X4',
        'results_dir': TRACK1_CROPPED_X4_FSRCNN_RESULTS_DIR
    }
]

TRACK2_FSRCNN_TARGETS = [
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X2) [Validation]',
        'scale': SCALE_X2,
        'weights_file': MODEL_FSRCNN_X2_WEIGHTS,
        'src_dir': PROCESSED_TRACK2_VALID_DIR + '/X2',
        'results_dir': TRACK2_CROPPED_X2_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X3) [Validation]',
        'scale': SCALE_X3,
        'weights_file': MODEL_FSRCNN_X3_WEIGHTS,
        'src_dir': PROCESSED_TRACK2_VALID_DIR + '/X3',
        'results_dir': TRACK2_CROPPED_X3_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X4) [Validation]',
        'scale': SCALE_X4,
        'weights_file': MODEL_FSRCNN_X4_WEIGHTS,
        'src_dir': PROCESSED_TRACK2_VALID_DIR + '/X4',
        'results_dir': TRACK2_CROPPED_X4_FSRCNN_RESULTS_DIR
    }
]

EVALUATION_TARGETS = [
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Original',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'scale': SCALE_X1, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'ESRGAN (ESRGAN)',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'ESRGAN (PSNR)',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X2_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X3_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X4_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X2_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X3_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': PROCESSED_TRACK1_VALID_DIR + '/HR', 
        'up_dir': TRACK1_CROPPED_X4_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Original',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'scale': SCALE_X1, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X3_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X4_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'ESRGAN (ESRGAN)',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X4_ESRGAN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'ESRGAN (PSNR)',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X2_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X3_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X4_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X2_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X3_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': PROCESSED_TRACK2_VALID_DIR + '/HR', 
        'up_dir': TRACK2_CROPPED_X4_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    }
]
