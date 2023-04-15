"""Test Constants Module"""

# TEST DATASET NAMES
SET5 = 'Set5'
SET14 = 'Set14'
URBAN100 = 'Urban100'
BSD100 = 'BSD100'

# IMG_TAGS
HR = 'HR'
LR = 'LR'
BICUBIC = 'bicubic'
GLASNER = 'glasner'
NEAREST = 'nearest'
KIM = 'Kim'
APLUS = 'A+'
SCSR = 'ScSR'
SELF_EX_SR = 'SelfExSR'
FSRCNN = 'FSRCNN'
FSRCNN_T1 = 'FSRCNNT1'
FSRCNN_T2 = 'FSRCNNT2'
SRCNN = 'SRCNN'
RRDBESRGAN = 'RRDBESRGAN'
RRDBPSNR = 'RRDBPSNR'

# DATASET ROOTS 
DATASET_DIR = './dataset'
SET5_ROOT = DATASET_DIR + '/' + SET5
SET14_ROOT = DATASET_DIR + '/' + SET14
URBAN100_ROOT = DATASET_DIR + '/' + URBAN100
BSD100_ROOT = DATASET_DIR + '/' + BSD100

# DATASET SUB PATHS
IMGAGE_SRF_X2 = '/image_SRF_2'
IMGAGE_SRF_X3 = '/image_SRF_3'
IMGAGE_SRF_X4 = '/image_SRF_4'
IMGAGE_SRF_X8 = '/image_SRF_8'

# MODELS
MODELS_DIR = './models'
MODEL_RRDB_ESRGAN_X4 = MODELS_DIR + '/RRDB_ESRGAN_x4.pth'
MODEL_RRDB_PSNR_X4 = MODELS_DIR + '/RRDB_PSNR_x4.pth'
MODEL_FSRCNN_X2_WEIGHTS = MODELS_DIR + '/fsrcnn_x2.pth'
MODEL_FSRCNN_X3_WEIGHTS = MODELS_DIR + '/fsrcnn_x3.pth'
MODEL_FSRCNN_X4_WEIGHTS = MODELS_DIR + '/fsrcnn_x4.pth'
MODEL_FSRCNN_T1_X4_WEIGHTS = MODELS_DIR + '/fsrcnn/track1/track1_x4__best.pth' 
MODEL_FSRCNN_T2_X4_WEIGHTS = MODELS_DIR + '/fsrcnn/track2/track2_x4__best.pth' 

# RESULTS ROOTS
RESULTS_DIR = './results'
SET5_RESULTS_ROOT = RESULTS_DIR + '/' + SET5
SET14_RESULTS_ROOT = RESULTS_DIR + '/' + SET14
URBAN100_RESULTS_ROOT = RESULTS_DIR + '/' + URBAN100
BSD100_RESULTS_ROOT = RESULTS_DIR + '/' + BSD100

# RESULTS SUB PATHS
RRDB_ESRGAN_DIR = '/' + RRDBESRGAN
RRDB_PSNR_DIR = '/' + RRDBPSNR
FSRCNN_DIR = '/' + FSRCNN
FSRCNN_T1_DIR = '/' + FSRCNN_T1
FSRCNN_T2_DIR = '/' + FSRCNN_T2


SCALE_X2 = 2 
SCALE_X3 = 3 
SCALE_X4 = 4 
SCALE_X8 = 8
PROGRESS_NUM = 10

PLOTS_DIR = './plots'
IMAGES_DIR = PLOTS_DIR + '/images'

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

TARGETS_FSRCNN = [
    {
        'dataset': SET5, 
        'src_dir': SET5_ROOT + IMGAGE_SRF_X2,
        'res_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X2 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X2,
        'weights': MODEL_FSRCNN_X2_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': SET5, 
        'src_dir': SET5_ROOT + IMGAGE_SRF_X3,
        'res_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X3 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X3,
        'weights': MODEL_FSRCNN_X3_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': SET5, 
        'src_dir': SET5_ROOT + IMGAGE_SRF_X4,
        'res_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_X4_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': SET14, 
        'src_dir': SET14_ROOT + IMGAGE_SRF_X2,
        'res_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X2 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X2,
        'weights': MODEL_FSRCNN_X2_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': SET14, 
        'src_dir': SET14_ROOT + IMGAGE_SRF_X3,
        'res_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X3 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X3,
        'weights': MODEL_FSRCNN_X3_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': SET14, 
        'src_dir': SET14_ROOT + IMGAGE_SRF_X4,
        'res_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_X4_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': URBAN100, 
        'src_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
        'res_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X2 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X2,
        'weights': MODEL_FSRCNN_X2_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': URBAN100, 
        'src_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
        'res_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_X4_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': BSD100, 
        'src_dir': BSD100_ROOT + IMGAGE_SRF_X2,
        'res_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X2 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X2,
        'weights': MODEL_FSRCNN_X2_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': BSD100, 
        'src_dir': BSD100_ROOT + IMGAGE_SRF_X3,
        'res_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X3 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X3,
        'weights': MODEL_FSRCNN_X3_WEIGHTS,
        'model': FSRCNN
    },
    {
        'dataset': BSD100, 
        'src_dir': BSD100_ROOT + IMGAGE_SRF_X4,
        'res_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_X4_WEIGHTS,
        'model': FSRCNN
    },



    {
        'dataset': SET5, 
        'src_dir': SET5_ROOT + IMGAGE_SRF_X4,
        'res_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T1_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_T1_X4_WEIGHTS,
        'model': FSRCNN_T1
    },
    {
        'dataset': SET14, 
        'src_dir': SET14_ROOT + IMGAGE_SRF_X4,
        'res_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T1_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_T1_X4_WEIGHTS,
        'model': FSRCNN_T1
    },
    {
        'dataset': URBAN100, 
        'src_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
        'res_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T1_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_T1_X4_WEIGHTS,
        'model': FSRCNN_T1
    },
    {
        'dataset': BSD100, 
        'src_dir': BSD100_ROOT + IMGAGE_SRF_X4,
        'res_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T1_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_T1_X4_WEIGHTS,
        'model': FSRCNN_T1
    },
    {
        'dataset': SET5, 
        'src_dir': SET5_ROOT + IMGAGE_SRF_X4,
        'res_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T2_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_T2_X4_WEIGHTS,
        'model': FSRCNN_T2
    },
    {
        'dataset': SET14, 
        'src_dir': SET14_ROOT + IMGAGE_SRF_X4,
        'res_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T2_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_T2_X4_WEIGHTS,
        'model': FSRCNN_T2
    },
    {
        'dataset': URBAN100, 
        'src_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
        'res_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T2_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_T2_X4_WEIGHTS,
        'model': FSRCNN_T2
    },
    {
        'dataset': BSD100, 
        'src_dir': BSD100_ROOT + IMGAGE_SRF_X4,
        'res_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T2_DIR,
        'src_tag': LR, 
        'scale': SCALE_X4,
        'weights': MODEL_FSRCNN_T2_X4_WEIGHTS,
        'model': FSRCNN_T2
    },
]

TARGETS_EVALUATION = [
    {
        'dataset': SET5, 
        'scale': SCALE_X2,
        'eval_file': f'./evaluation/{SET5}_X{SCALE_X2}_eval.csv',
        'hr_dir': SET5_ROOT + IMGAGE_SRF_X2,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': GLASNER, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': NEAREST, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SRCNN, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SCSR, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': FSRCNN, 
                'up_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X2 + FSRCNN_DIR,
            }
        ]
    },
    {
        'dataset': SET5, 
        'scale': SCALE_X3,
        'eval_file': f'./evaluation/{SET5}_X{SCALE_X3}_eval.csv',
        'hr_dir': SET5_ROOT + IMGAGE_SRF_X3,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': GLASNER, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': NEAREST, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SRCNN, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SCSR, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': FSRCNN, 
                'up_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X3 + FSRCNN_DIR,
            }
        ]
    },
    {
        'dataset': SET5, 
        'scale': SCALE_X4,
        'eval_file': f'./evaluation/{SET5}_X{SCALE_X4}_eval.csv',
        'hr_dir': SET5_ROOT + IMGAGE_SRF_X4,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': GLASNER, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': NEAREST, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SRCNN, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SCSR, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': SET5_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': FSRCNN, 
                'up_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_DIR,
            },
            {
                'tag': RRDBESRGAN, 
                'up_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4 + RRDB_ESRGAN_DIR,
            },
            {
                'tag': RRDBPSNR, 
                'up_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4 + RRDB_PSNR_DIR,
            },
            {
                'tag': FSRCNN_T1, 
                'up_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T1_DIR,
            },
            {
                'tag': FSRCNN_T2, 
                'up_dir': SET5_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T2_DIR,
            }
            
        ]
    },
    {
        'dataset': SET14, 
        'scale': SCALE_X2,
        'eval_file': f'./evaluation/{SET14}_X{SCALE_X2}_eval.csv',
        'hr_dir': SET14_ROOT + IMGAGE_SRF_X2,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': GLASNER, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': NEAREST, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SRCNN, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SCSR, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': FSRCNN, 
                'up_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X2 + FSRCNN_DIR,
            }
        ]
    },
    {
        'dataset': SET14, 
        'scale': SCALE_X3,
        'eval_file': f'./evaluation/{SET14}_X{SCALE_X3}_eval.csv',
        'hr_dir': SET14_ROOT + IMGAGE_SRF_X3,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': GLASNER, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': NEAREST, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SRCNN, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SCSR, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': FSRCNN, 
                'up_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X3 + FSRCNN_DIR,
            }
        ]
    },
    {
        'dataset': SET14, 
        'scale': SCALE_X4,
        'eval_file': f'./evaluation/{SET14}_X{SCALE_X4}_eval.csv',
        'hr_dir': SET14_ROOT + IMGAGE_SRF_X4,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': GLASNER, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': NEAREST, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SRCNN, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SCSR, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': SET14_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': FSRCNN, 
                'up_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_DIR,
            },
            {
                'tag': RRDBESRGAN, 
                'up_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4 + RRDB_ESRGAN_DIR,
            },
            {
                'tag': RRDBPSNR, 
                'up_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4 + RRDB_PSNR_DIR,
            },
            {
                'tag': FSRCNN_T1, 
                'up_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T1_DIR,
            },
            {
                'tag': FSRCNN_T2, 
                'up_dir': SET14_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T2_DIR,
            }
        ]
    },
    {
        'dataset': URBAN100, 
        'scale': SCALE_X2,
        'eval_file': f'./evaluation/{URBAN100}_X{SCALE_X2}_eval.csv',
        'hr_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': GLASNER, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': NEAREST, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SRCNN, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': APLUS, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SCSR, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': FSRCNN, 
                'up_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X2 + FSRCNN_DIR,
            },
        ]
    },
    {
        'dataset': URBAN100, 
        'scale': SCALE_X4,
        'eval_file': f'./evaluation/{URBAN100}_X{SCALE_X4}_eval.csv',
        'hr_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': GLASNER, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': NEAREST, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SRCNN, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SCSR, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': APLUS, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': URBAN100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': FSRCNN, 
                'up_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_DIR,
            },
            {
                'tag': RRDBESRGAN, 
                'up_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4 + RRDB_ESRGAN_DIR,
            },
            {
                'tag': RRDBPSNR, 
                'up_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4 + RRDB_PSNR_DIR,
            },
            {
                'tag': FSRCNN_T1, 
                'up_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T1_DIR,
            },
            {
                'tag': FSRCNN_T2, 
                'up_dir': URBAN100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T2_DIR,
            }
        ]
    },
    {
        'dataset': BSD100, 
        'scale': SCALE_X2,
        'eval_file': f'./evaluation/{BSD100}_X{SCALE_X2}_eval.csv',
        'hr_dir': BSD100_ROOT + IMGAGE_SRF_X2,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': GLASNER, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': NEAREST, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SRCNN, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SCSR, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': APLUS, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X2,
            },
            {
                'tag': FSRCNN, 
                'up_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X2 + FSRCNN_DIR,
            }
        ]
    },
    {
        'dataset': BSD100, 
        'scale': SCALE_X3,
        'eval_file': f'./evaluation/{BSD100}_X{SCALE_X3}_eval.csv',
        'hr_dir': BSD100_ROOT + IMGAGE_SRF_X3,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': GLASNER, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': NEAREST, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SRCNN, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SCSR, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': APLUS, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X3,
            },
            {
                'tag': FSRCNN, 
                'up_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X3 + FSRCNN_DIR,
            }
        ]
    },
    {
        'dataset': BSD100, 
        'scale': SCALE_X4,
        'eval_file': f'./evaluation/{BSD100}_X{SCALE_X4}_eval.csv',
        'hr_dir': BSD100_ROOT + IMGAGE_SRF_X4,
        'models': [
            {
                'tag': BICUBIC, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': GLASNER, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': NEAREST, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SRCNN, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': APLUS, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SCSR, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': SELF_EX_SR, 
                'up_dir': BSD100_ROOT + IMGAGE_SRF_X4,
            },
            {
                'tag': FSRCNN, 
                'up_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_DIR,
            },
            {
                'tag': RRDBESRGAN, 
                'up_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4 + RRDB_ESRGAN_DIR,
            },
            {
                'tag': RRDBPSNR, 
                'up_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4 + RRDB_PSNR_DIR,
            },
            {
                'tag': FSRCNN_T1, 
                'up_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T1_DIR,
            },
            {
                'tag': FSRCNN_T2, 
                'up_dir': BSD100_RESULTS_ROOT + IMGAGE_SRF_X4 + FSRCNN_T2_DIR,
            }            
        ]
    },
]