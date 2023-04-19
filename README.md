# AMLS_II_assignment

__Applied Machine Learning Systems II (ELEC0135) 22/23 Assignment Code__

_Abstract_: Single-image super-resolution (SR) is an important computer vision challenge that aims to reconstruct high-resolution images from low-resolution ones. Various statistical algorithms were employed for this purpose until deep learning models could be used to address this in a much more efficient manner. This report details the research and deep learning models that were used to evaluate their SR performance against benchmark datasets, using common image quality assessment (IQA) metrics such as peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), and Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) score. 

There are various techniques that can be used for image super-resolution. These range from classical interpolation-based statistical algorithms to deep learning approaches which employ models such as convolutional neural networks (CNNs) and generative adversarial networks (GANs). This assignment aims to assess some of these single-image SR models using popular benchmark datasets for training and evaluation.

## Instructions to Run the Project Code

### Creating the Conda Environment
The conda environment can be created using the [environment.yml](./environment.yml) file provided. The following commands can be run in order to create and access the environment. 

```
conda env create -f environment.yml
conda activate amls2
```

### Getting the Test Data
Even though the code downloads the DIV2K dataset, the testing datasets need to be download and placed in the [dataset](./dataset) folder. Refer to the [examples](./examples) folder to check the folder format of the test datasets. Test datasets can be downloaded from the following location. 

- [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
- [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
- [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip)
- [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)

### Running the Code
Once in the conda environment, the entire pipeline can be executed from [main.py](./main.py), using the following command.

```
python main.py
```

_It is important to note that a large amount of disk space is required in order to run the entire pipeline (Upwards of 120GB (estimated)). A large RAM, and a good quality GPU will also be required. It is hard to estimate the run time of the entire pipeline, but it is expected to take multiple hours_

## Project Structure

 * [main.py](./main.py) - Runs the entire pipeline for the project
 * [constants.py](./constants.py) - Contains constants used in the project
 * [constants2.py](./constants2.py) - Contains some more constants used in the project
 * [modules](./modules) - directory containing various modules 
   * [bicubic.py](./modules/bicubic.py) - Provides functionality to perform bicubic interpolation
   * [data_processing.py](./modules/data_processing.py) - Performs data processing functions such as cropping the images and creating .h5 datasets for training 
   * [dataset.py](./modules/dataset.py) - Downloads the DIV2K dataset, and provides dataset loader objects for training the FSRCNN model
   * [ESRGAN.py](./modules/ESRGAN.py) - Implements the ESRGAN model using the RRDB architecture
   * [evaluation.py](./modules/evaluation.py) - Contains various functions for evaluating the results of the models using the upscaled images
   * [FSRCNN.py](./modules/FSRCNN.py) - Implements the FSRCNN model
   * [plotter.py](./modules/plotter.py) - Includes various functions to plot charts and images
   * [RRDB_Net.py](./modules/RRDB_Net.py) - Implements the RRDB architecture. Used for the ESRGAN model
   * [run_models.py](./modules/run_models.py) - Runs all the models in the project on the test datasets (Set5, Set100, Urban100, BSD100)
   * [runner.py](./modules/runner.py) - Runs the models on the DIV2K validation datasets. Deprecated.
   * [SRCNN.py](./modules/SRCNN.py) - Implements the SRCNN model. 
   * [train_FSRCNN.py](./modules/train_FSRCNN.py) - Trains various versions of the FSRCNN model from the DIV2K training data (and validates with the validation data) - all 3 scaling factors and both tracks
   * [.py](./modules/.py)
 * [helpers](./helpers) - Directory containing common functions used by multiple classes 
   * [helpers.py](./helpers/helpers.py) - Contains various helper functions used by different models
   * [utility.py](./helpers/utility.py) - Provides common IO functions used across the project
 * [Track 1](./track1) - Modules for track 1. _Deprecated_.
   * [track1.py](./track1/track1.py) - Runs models on the Track 1 (NTIRE 2017) validation  data. _Deprecated_.
 * [Track 2](./track2) - Modules for track 2. _Deprecated_.
   * [track2.py](./track2/track2.py) - Runs models on the Track 2 (NTIRE 2017) validation  data. _Deprecated_.
 * [models](./models) - Directory containing pre-trained model weights
 * [plots](./plots) - Directory containing generated plots and images
 * [examples](./examples) - Directory containing samples of other directories, not included on GitHub.
   * [dataset](./examples/dataset) - Directory containing the datasets.
   * [processed](./examples/processed) - Directory containing the processed data. The processed .h5 files are not included due to their size.
   * [results](./examples/results) - Directory containing the upscaled image results.
 * [evaluation](./evaluation) - Directory containing evaluation results 
 * [environment.yml](./environment.yml) - File used to create the conda environment. Contains information about dependencies used in the project.
 * [README.md](./README.md) - This file :)
 * [.gitignore](./.gitignore) - Git ignore
 * [.gitattributes](./.gitattributes) - Git attributes

### Other Directories (Not included on GitHub)
 * [dataset](./dataset) - Directory containing the datasets. Not included on github. Gets created during code execution.
 * [processed](./processed) - Directory containing the processed data. Not included on github. Gets created during code execution.
 * [results](./results) - Directory containing the upscaled image results. Not included on github. Gets created during code execution.

## Code References

Code from the following sources has been used in this project to implement various models and algorithms. The places where this code has been used have been highlighted in the code itself.

- https://www.geeksforgeeks.org/python-opencv-bicubic-interpolation-for-resizing-image/
- https://github.com/MarkPrecursor/SRCNN-keras
- https://github.com/yjn870/FSRCNN-pytorch
- https://github.com/xinntao/ESRGAN