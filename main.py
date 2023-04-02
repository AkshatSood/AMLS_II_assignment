"""Main Module"""

from modules.data_processing import DataProcessing
from modules.evaluation import Evaluation
from track1.track1 import Track1
from track2.track2 import Track2


def main():

    # # Process Data
    # print('\n#########################################')
    # print('\tDATA PROCESSING')
    # print('#########################################\n')
    # data_processor = DataProcessing()
    # data_processor.process()

    # Perform Track 1 Tasks
    print('\n#########################################')
    print('\tTRACK 1 TASKS')
    print('#########################################\n')
    track1 = Track1()
    track1.run(
        run_bicubic_interpolation=False,
        run_real_esrgan=True,
        run_srcnn=False,
        run_fsrcnn=False
    )

    # Perform Track 2 Tasks
    print('\n#########################################')
    print('\tTRACK 2 TASKS')
    print('#########################################\n')
    track2 = Track2()
    track2.run(
        run_bicubic_interpolation=False,
        run_real_esrgan=True,
        run_srcnn=False,
        run_fsrcnn=False
    )
    
    # Evaluate Data
    print('\n#########################################')
    print('\tEVALUATION')
    print('#########################################\n')
    evaluation = Evaluation()
    evaluation.evaluate()

if __name__ == "__main__":
    main()
