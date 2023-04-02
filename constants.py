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
TRACK1_PROCESSED_TRAIN_X3_DIR = PROCESSED_TRACK1_TRAIN_DIR + '/X3'
TRACK1_PROCESSED_TRAIN_X4_DIR = PROCESSED_TRACK1_TRAIN_DIR + '/X4'

TRACK1_PROCESSED_VALID_HR_DIR = PROCESSED_TRACK1_VALID_DIR + '/HR'
TRACK1_PROCESSED_VALID_X2_DIR = PROCESSED_TRACK1_VALID_DIR + '/X2'
TRACK1_PROCESSED_VALID_X3_DIR = PROCESSED_TRACK1_VALID_DIR + '/X3'
TRACK1_PROCESSED_VALID_X4_DIR = PROCESSED_TRACK1_VALID_DIR + '/X4'

PROCESSED_TRACK2_TRAIN_DIR = PROCESSED_TRACK2 + '/train'
PROCESSED_TRACK2_VALID_DIR = PROCESSED_TRACK2 + '/valid'

TRACK2_PROCESSED_TRAIN_HR_DIR = PROCESSED_TRACK2_TRAIN_DIR + '/HR'
TRACK2_PROCESSED_TRAIN_X2_DIR = PROCESSED_TRACK2_TRAIN_DIR + '/X2'
TRACK2_PROCESSED_TRAIN_X3_DIR = PROCESSED_TRACK2_TRAIN_DIR + '/X3'
TRACK2_PROCESSED_TRAIN_X4_DIR = PROCESSED_TRACK2_TRAIN_DIR + '/X4'

TRACK2_PROCESSED_VALID_HR_DIR = PROCESSED_TRACK2_VALID_DIR + '/HR'
TRACK2_PROCESSED_VALID_X2_DIR = PROCESSED_TRACK2_VALID_DIR + '/X2'
TRACK2_PROCESSED_VALID_X3_DIR = PROCESSED_TRACK2_VALID_DIR + '/X3'
TRACK2_PROCESSED_VALID_X4_DIR = PROCESSED_TRACK2_VALID_DIR + '/X4'

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
TRACK1_CROPPED_REAL_ESRGAN_RESULTS_DIR = TRACK1_CROPPED_RESULTS_DIR + '/real_esrgan_esrgan'
TRACK2_CROPPED_REAL_ESRGAN_RESULTS_DIR = TRACK2_CROPPED_RESULTS_DIR + '/real_esrgan_esrgan'
TRACK1_CROPPED_REAL_ESRGAN_PSNR_RESULTS_DIR = TRACK1_CROPPED_RESULTS_DIR + '/real_esrgan_psnr'
TRACK2_CROPPED_REAL_ESRGAN_PSNR_RESULTS_DIR = TRACK2_CROPPED_RESULTS_DIR + '/real_esrgan_psnr'

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

TRACK1_X4_ESRGAN_RESULTS_DIR = TRACK1_REAL_ESRGAN_RESULTS_DIR + '/X4'
TRACK2_X4_ESRGAN_RESULTS_DIR = TRACK2_REAL_ESRGAN_RESULTS_DIR + '/X4'
TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR = TRACK1_CROPPED_REAL_ESRGAN_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_ESRGAN_RESULTS_DIR = TRACK2_CROPPED_REAL_ESRGAN_RESULTS_DIR + '/X4'
TRACK1_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR = TRACK1_CROPPED_REAL_ESRGAN_PSNR_RESULTS_DIR + '/X4'
TRACK2_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR = TRACK2_CROPPED_REAL_ESRGAN_PSNR_RESULTS_DIR + '/X4'

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

# TARGETS
DATA_PROCESSING_TARGETS = [
    {
        'name': 'Track 1 - Training Data (HR)',
        'src_dir': TRAIN_HR,
        'output_dir': TRACK1_PROCESSED_TRAIN_HR_DIR,
        'scale': SCALE_X1
    },
    {
        'name': 'Track 1 - Training Data (X2)',
        'src_dir': TRACK1_X2_TRAIN,
        'output_dir': TRACK1_PROCESSED_TRAIN_X2_DIR,
        'scale': SCALE_X2
    },
    {
        'name': 'Track 1 - Training Data (X3)',
        'src_dir': TRACK1_X3_TRAIN,
        'output_dir': TRACK1_PROCESSED_TRAIN_X3_DIR,
        'scale': SCALE_X3
    },
    {
        'name': 'Track 1 - Training Data (X4)',
        'src_dir': TRACK1_X4_TRAIN,
        'output_dir': TRACK1_PROCESSED_TRAIN_X4_DIR,
        'scale': SCALE_X4
    },
    {
        'name': 'Track 1 - Validation Data (HR)',
        'src_dir': VALIDATION_HR,
        'output_dir': TRACK1_PROCESSED_VALID_HR_DIR,
        'scale': SCALE_X1
    },
    {
        'name': 'Track 1 - Validation Data (X2)',
        'src_dir': TRACK1_X2_VALIDATION,
        'output_dir': TRACK1_PROCESSED_VALID_X2_DIR,
        'scale': SCALE_X2
    },
    {
        'name': 'Track 1 - Validation Data (X3)',
        'src_dir': TRACK1_X3_VALIDATION,
        'output_dir': TRACK1_PROCESSED_VALID_X3_DIR,
        'scale': SCALE_X3
    },
    {
        'name': 'Track 1 - Validation Data (X4)',
        'src_dir': TRACK1_X4_VALIDATION,
        'output_dir': TRACK1_PROCESSED_VALID_X4_DIR,
        'scale': SCALE_X4
    },
    {
        'name': 'Track 2 - Training Data (HR)',
        'src_dir': TRAIN_HR,
        'output_dir': TRACK2_PROCESSED_TRAIN_HR_DIR,
        'scale': SCALE_X1
    },
    {
        'name': 'Track 2 - Training Data (X2)',
        'src_dir': TRACK2_X2_TRAIN,
        'output_dir': TRACK2_PROCESSED_TRAIN_X2_DIR,
        'scale': SCALE_X2
    },
    {
        'name': 'Track 2 - Training Data (X3)',
        'src_dir': TRACK2_X3_TRAIN,
        'output_dir': TRACK2_PROCESSED_TRAIN_X3_DIR,
        'scale': SCALE_X3
    },
    {
        'name': 'Track 2 - Training Data (X4)',
        'src_dir': TRACK2_X4_TRAIN,
        'output_dir': TRACK2_PROCESSED_TRAIN_X4_DIR,
        'scale': SCALE_X4
    },
    {
        'name': 'Track 2 - Validation Data (HR)',
        'src_dir': VALIDATION_HR,
        'output_dir': TRACK2_PROCESSED_VALID_HR_DIR,
        'scale': SCALE_X1
    },
    {
        'name': 'Track 2 - Validation Data (X2)',
        'src_dir': TRACK2_X2_VALIDATION,
        'output_dir': TRACK2_PROCESSED_VALID_X2_DIR,
        'scale': SCALE_X2
    },
    {
        'name': 'Track 2 - Validation Data (X3)',
        'src_dir': TRACK2_X3_VALIDATION,
        'output_dir': TRACK2_PROCESSED_VALID_X3_DIR,
        'scale': SCALE_X3
    },
    {
        'name': 'Track 2 - Validation Data (X4)',
        'src_dir': TRACK2_X4_VALIDATION,
        'output_dir': TRACK2_PROCESSED_VALID_X4_DIR,
        'scale': SCALE_X4
    },
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
        'src_dir': TRACK1_PROCESSED_VALID_X2_DIR,
        'results_dir': TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 (Cropped) - Bicubic Interpolation (x3) [Validation]',
        'scale': SCALE_X3,
        'src_dir': TRACK1_PROCESSED_VALID_X3_DIR,
        'results_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 (Cropped) - Bicubic Interpolation (x4) [Validation]',
        'scale': SCALE_X4,
        'src_dir': TRACK1_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 (Cropped) - Bicubic Interpolation (x2) [Train]',
        'scale': SCALE_X2,
        'src_dir': TRACK1_PROCESSED_TRAIN_X2_DIR,
        'results_dir': TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 (Cropped) - Bicubic Interpolation (x3) [Train]',
        'scale': SCALE_X3,
        'src_dir': TRACK1_PROCESSED_TRAIN_X3_DIR,
        'results_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 1 (Cropped) - Bicubic Interpolation (x4) [Train]',
        'scale': SCALE_X4,
        'src_dir': TRACK1_PROCESSED_TRAIN_X4_DIR,
        'results_dir': TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR,
    }
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
        'src_dir': TRACK2_PROCESSED_VALID_X2_DIR,
        'results_dir': TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 (Cropped) - Bicubic Interpolation (x3) [Validation]',
        'scale': SCALE_X3,
        'src_dir': TRACK2_PROCESSED_VALID_X3_DIR,
        'results_dir': TRACK2_CROPPED_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 (Cropped) - Bicubic Interpolation (x4) [Validation]',
        'scale': SCALE_X4,
        'src_dir': TRACK2_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK2_CROPPED_X4_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 (Cropped) - Bicubic Interpolation (x2) [Train]',
        'scale': SCALE_X2,
        'src_dir': TRACK2_PROCESSED_TRAIN_X2_DIR,
        'results_dir': TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 (Cropped) - Bicubic Interpolation (x3) [Train]',
        'scale': SCALE_X3,
        'src_dir': TRACK2_PROCESSED_TRAIN_X3_DIR,
        'results_dir': TRACK2_CROPPED_X3_BICUBIC_RESULTS_DIR,
    },
    {
        'name': 'Track 2 (Cropped) - Bicubic Interpolation (x4) [Train]',
        'scale': SCALE_X4,
        'src_dir': TRACK2_PROCESSED_TRAIN_X4_DIR,
        'results_dir': TRACK2_CROPPED_X4_BICUBIC_RESULTS_DIR,
    }
]

TRACK1_ESRGANX4_TARGETS = [
    # {
    #     'name': 'Track 1 - Real-ESRGAN (X4)',
    #     'scale': SCALE_X4,
    #     'src_dir': TRACK1_X4_VALIDATION,
    #     'results_dir': TRACK1_X4_ESRGAN_RESULTS_DIR
    # },
    {
        'name': 'Track 1 (Cropped) - Real-ESRGAN (ESRGAN) (X4) [Validation]',
        'scale': SCALE_X4,
        'model': MODEL_RRDB_ESRGAN_X4,
        'src_dir': TRACK1_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR
    },
    # {
    #     'name': 'Track 1 (Cropped) - Real-ESRGAN (ESRGAN) (X4) [Train]',
    #     'scale': SCALE_X4,
    #     'model': MODEL_RRDB_ESRGAN_X4,
    #     'src_dir': TRACK1_PROCESSED_TRAIN_X4_DIR,
    #     'results_dir': TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR
    # },
    {
        'name': 'Track 1 (Cropped) - Real-ESRGAN (PSNR) (X4) [Validation]',
        'scale': SCALE_X4,
        'model': MODEL_RRDB_PSNR_X4,
        'src_dir': TRACK1_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK1_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR
    },
    # {
    #     'name': 'Track 1 (Cropped) - Real-ESRGAN (PSNR) (X4) [Train]',
    #     'scale': SCALE_X4,
    #     'model': MODEL_RRDB_PSNR_X4,
    #     'src_dir': TRACK1_PROCESSED_TRAIN_X4_DIR,
    #     'results_dir': TRACK1_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR
    # }
]

TRACK2_ESRGANX4_TARGETS = [
    # {
    #     'name': 'Track 2 - Real-ESRGAN (x4)',
    #     'scale': SCALE_X4,
    #     'src_dir': TRACK2_X4_VALIDATION,
    #     'results_dir': TRACK2_X4_ESRGAN_RESULTS_DIR
    # },
    {
        'name': 'Track 2 (Cropped) - Real-ESRGAN (ESRGAN) (x4) [Validation]',
        'scale': SCALE_X4,
        'model': MODEL_RRDB_ESRGAN_X4,
        'src_dir': TRACK2_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK2_CROPPED_X4_ESRGAN_RESULTS_DIR
    },
    # {
    #     'name': 'Track 2 (Cropped) - Real-ESRGAN (ESRGAN) (x4) [Train]',
    #     'scale': SCALE_X4,
    #     'model': MODEL_RRDB_ESRGAN_X4,
    #     'src_dir': TRACK2_PROCESSED_TRAIN_X4_DIR,
    #     'results_dir': TRACK2_CROPPED_X4_ESRGAN_RESULTS_DIR
    # },
    {
        'name': 'Track 2 (Cropped) - Real-ESRGAN (PSNR) (x4) [Validation]',
        'scale': SCALE_X4,
        'model': MODEL_RRDB_PSNR_X4,
        'src_dir': TRACK2_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK2_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR
    },
    # {
    #     'name': 'Track 2 (Cropped) - Real-ESRGAN (PSNR) (x4) [Train]',
    #     'scale': SCALE_X4,
    #     'model': MODEL_RRDB_PSNR_X4,
    #     'src_dir': TRACK2_PROCESSED_TRAIN_X4_DIR,
    #     'results_dir': TRACK2_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR
    # }
]

TRACK1_SRCNN_TARGETS = [
    {
        'name': 'Track 1 (Cropped) - SRCNN (X2) [Validation and Train]',
        'scale': SCALE_X2,
        'border': SRCNN_X2_BORDER,
        'src_dir': TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK1_CROPPED_X2_SRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - SRCNN (X3) [Validation and Train]',
        'scale': SCALE_X3,
        'border': SRCNN_X3_BORDER,
        'src_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK1_CROPPED_X3_SRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - SRCNN (X4) [Validation and Train]',
        'scale': SCALE_X4,
        'border': SRCNN_X4_BORDER,
        'src_dir': TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK1_CROPPED_X4_SRCNN_RESULTS_DIR
    }
]

TRACK2_SRCNN_TARGETS = [
    {
        'name': 'Track 2 (Cropped) - SRCNN (X2) [Validation and Train]',
        'scale': SCALE_X2,
        'border': SRCNN_X2_BORDER,
        'src_dir': TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK2_CROPPED_X2_SRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - SRCNN (X3) [Validation and Train]',
        'scale': SCALE_X3,
        'border': SRCNN_X3_BORDER,
        'src_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR,
        'results_dir': TRACK2_CROPPED_X3_SRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - SRCNN (X4) [Validation and Train]',
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
        'src_dir': TRACK1_PROCESSED_VALID_X2_DIR,
        'results_dir': TRACK1_CROPPED_X2_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - FSRCNN (X3) [Validation]',
        'scale': SCALE_X3,
        'weights_file': MODEL_FSRCNN_X3_WEIGHTS,
        'src_dir': TRACK1_PROCESSED_VALID_X3_DIR,
        'results_dir': TRACK1_CROPPED_X3_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - FSRCNN (X4) [Validation]',
        'scale': SCALE_X4,
        'weights_file': MODEL_FSRCNN_X4_WEIGHTS,
        'src_dir': TRACK1_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK1_CROPPED_X4_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - FSRCNN (X2) [Train]',
        'scale': SCALE_X2,
        'weights_file': MODEL_FSRCNN_X2_WEIGHTS,
        'src_dir': TRACK1_PROCESSED_TRAIN_X2_DIR,
        'results_dir': TRACK1_CROPPED_X2_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - FSRCNN (X3) [Train]',
        'scale': SCALE_X3,
        'weights_file': MODEL_FSRCNN_X3_WEIGHTS,
        'src_dir': TRACK1_PROCESSED_TRAIN_X3_DIR,
        'results_dir': TRACK1_CROPPED_X3_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 1 (Cropped) - FSRCNN (X4) [Train]',
        'scale': SCALE_X4,
        'weights_file': MODEL_FSRCNN_X4_WEIGHTS,
        'src_dir': TRACK1_PROCESSED_TRAIN_X4_DIR,
        'results_dir': TRACK1_CROPPED_X4_FSRCNN_RESULTS_DIR
    },
]

TRACK2_FSRCNN_TARGETS = [
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X2) [Validation]',
        'scale': SCALE_X2,
        'weights_file': MODEL_FSRCNN_X2_WEIGHTS,
        'src_dir': TRACK2_PROCESSED_VALID_X2_DIR,
        'results_dir': TRACK2_CROPPED_X2_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X3) [Validation]',
        'scale': SCALE_X3,
        'weights_file': MODEL_FSRCNN_X3_WEIGHTS,
        'src_dir': TRACK2_PROCESSED_VALID_X3_DIR,
        'results_dir': TRACK2_CROPPED_X3_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X4) [Validation]',
        'scale': SCALE_X4,
        'weights_file': MODEL_FSRCNN_X4_WEIGHTS,
        'src_dir': TRACK2_PROCESSED_VALID_X4_DIR,
        'results_dir': TRACK2_CROPPED_X4_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X2) [Train]',
        'scale': SCALE_X2,
        'weights_file': MODEL_FSRCNN_X2_WEIGHTS,
        'src_dir': TRACK2_PROCESSED_TRAIN_X2_DIR,
        'results_dir': TRACK2_CROPPED_X2_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X3) [Train]',
        'scale': SCALE_X3,
        'weights_file': MODEL_FSRCNN_X3_WEIGHTS,
        'src_dir': TRACK2_PROCESSED_TRAIN_X3_DIR,
        'results_dir': TRACK2_CROPPED_X3_FSRCNN_RESULTS_DIR
    },
    {
        'name': 'Track 2 (Cropped) - FSRCNN (X4) [Train]',
        'scale': SCALE_X4,
        'weights_file': MODEL_FSRCNN_X4_WEIGHTS,
        'src_dir': TRACK2_PROCESSED_TRAIN_X4_DIR,
        'results_dir': TRACK2_CROPPED_X4_FSRCNN_RESULTS_DIR
    },
]

EVALUATION_TARGETS = [
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Original',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'scale': SCALE_X1, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X2_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X3_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X4_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Real-ESRGAN (ESRGAN)',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X4_ESRGAN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'Real-ESRGAN (PSNR)',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X2_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X3_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X4_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X2_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X3_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 1 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': TRACK1_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK1_CROPPED_X4_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Original',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'scale': SCALE_X1, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X2_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X3_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Bicubic',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X4_BICUBIC_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Real-ESRGAN (ESRGAN)',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X4_ESRGAN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'Real-ESRGAN (PSNR)',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X4_ESRGAN_PSNR_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X2_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X3_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'SRCNN',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X4_SRCNN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X2_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X2, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X3_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X3, 
        'startswith': tuple(['08', '09']),
    },
    {      
        'track': 'Track 2 (Cropped)', 
        'method': 'FSRCNN',
        'hr_dir': TRACK2_PROCESSED_VALID_HR_DIR, 
        'up_dir': TRACK2_CROPPED_X4_FSRCNN_RESULTS_DIR, 
        'scale': SCALE_X4, 
        'startswith': tuple(['08', '09']),
    }
]
