from track1.track1 import Track1
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
        run_bicubic_interpolation=False, 
        run_real_esrgan=True
    )

    # Perform Track 2 Tasks
    print('\n#########################################')
    print('\tTRACK 2 TASKS')
    print('#########################################\n')
    

if __name__ == "__main__":
    main()
