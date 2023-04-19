"""Main Module"""

from modules.data_processing import DataProcessing
from modules.dataset import Dataset
from modules.evaluation import Evaluation
from modules.plotter import Plotter
from modules.run_models import RunModels
from modules.train_FSRCNN import FSRCNNTrainer
from track1.track1 import Track1
from track2.track2 import Track2


def main():
    """This runs the entire pipeline for the code. 
    """
    # Acquire Data 
    # Downloads the DIV2K dataset to the ./dataset folder (unless its updated in the constants.py)
    print('\n#########################################')
    print('\tDATA ACQUISITION')
    print('#########################################\n')
    dataset = Dataset()
    dataset.download()

    # Process Data
    # Performs 2 forms of processing - creates cropped datasets from the DIV2K images, and 
    # creates .h5 files for FSRCNN model training and validation
    print('\n#########################################')
    print('\tDATA PROCESSING')
    print('#########################################\n')
    data_processor = DataProcessing()
    data_processor.crop_images()
    data_processor.create_training_datasets()

    # Train FSRCNN models
    # Trains the FSRCNN model with the .h5 files created in the previous step
    print('\n#########################################')
    print('\tTRAIN FSRCNN MODEL')
    print('#########################################\n')
    trainer = FSRCNNTrainer()
    trainer.train_models()
    
    
    # # Perform Track 1 Tasks
    # # Deprecated - results are not used in the report
    # print('\n#########################################')
    # print('\tTRACK 1 TASKS')
    # print('#########################################\n')
    # track1 = Track1()
    # track1.run(
    #     run_bicubic_interpolation=True,
    #     run_esrgan=True,
    #     run_srcnn=False,
    #     run_fsrcnn=False
    # )

    # # Perform Track 2 Tasks
    # # Deprecated - results are not used in the report
    # print('\n#########################################')
    # print('\tTRACK 2 TASKS')
    # print('#########################################\n')
    # track2 = Track2()
    # track2.run(
    #     run_bicubic_interpolation=True,
    #     run_esrgan=True,
    #     run_srcnn=False,
    #     run_fsrcnn=False
    # )

    # Run the models
    # Runs all the models (trained and pre-trained) on the test datasets
    print('\n#########################################')
    print('\tRUN MODELS')
    print('#########################################\n')
    model_runner = RunModels()
    model_runner.run_rrdb_esrgan_model()
    model_runner.run_rrdb_psnr_model()
    model_runner.run_fsrcnn_model()
    
    # Evaluate Data
    # Uses the results produced from the models in the previous step to evaluate 
    # the models
    print('\n#########################################')
    print('\tEVALUATION')
    print('#########################################\n')
    evaluation = Evaluation()
    evaluation.evaluate_tests()
    evaluation.create_evaluation_summary()

    # Plot images and charts
    # Used to plot various charts and images used in the report
    print('\n#########################################')
    print('\tPLOT IMAGES AND CHARTS')
    print('#########################################\n')
    plotter = Plotter()
    plotter.plot_zoomed_imgs()
    plotter.plot_epoch_psnr_charts()
    plotter.plot_summary_charts()
    plotter.plot_evaluation_charts()

if __name__ == "__main__":
    main()
