from track1.track1 import Track1
from track2.track2 import Track2
from modules.data_processing import DataProcessing

def main():

    # Process Data
    print('\n#########################################')
    print('\tDATA PROCESSING')
    print('#########################################\n')
    data_processor = DataProcessing()
    data_processor.process()

    # Perform Track 1 Tasks
    print('\n#########################################')
    print('\tTRACK 1 TASKS')
    print('#########################################\n')
    track1 = Track1()
    track1.run(
        run_bicubic_interpolation=True, 
        run_real_esrgan=False, 
        run_srcnn=True,
        run_fsrcnn = True
    )

    # Perform Track 2 Tasks
    print('\n#########################################')
    print('\tTRACK 2 TASKS')
    print('#########################################\n')
    track2 = Track2()
    track2.run(
        run_bicubic_interpolation=True,
        run_real_esrgan=False, 
        run_srcnn=True,
        run_fsrcnn = True
    )

if __name__ == "__main__":
    main()
