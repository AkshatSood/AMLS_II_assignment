"""Plotter Module"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cbook

from constants2 import IMAGES_DIR, PLOTS_DIR, TARGETS_EVALUATION
from helpers.helpers import Helpers
from helpers.utility import Utility


class Plotter: 

    def __init__(self):
        self.utility = Utility()
        self.helpers = Helpers()

        self.utility.check_and_create_dir(PLOTS_DIR)
        self.utility.check_and_create_dir(IMAGES_DIR)

        self.image_index = 2

    def ___plot_zoomed_img(self, img_path, output_path):

        img = cv2.imread(img_path)

        # Crop the image to a square
        h, w, c = img.shape
        if h < w:
            img = self.helpers.crop_to_shape(img, (h, h))
        else:
            img = self.helpers.crop_to_shape(img, (w, w))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0)

        extent=(-3, 4, -4, 3)

        fig, ax = plt.subplots(figsize=[5, 4])

        ax.imshow(img, extent=extent, origin="lower")

        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        axins.imshow(img, extent=extent, origin="lower")
        # subregion of the original image
        x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.set_xticks([])
        axins.set_yticks([])

        ax.indicate_inset_zoom(axins, edgecolor="black", lw=2)

        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        plt.savefig(output_path, bbox_inches='tight', dpi=600)

    def plot_zoomed_imgs(self):
        print('\n=> Plotting zoomed images')
        
        for target in TARGETS_EVALUATION:
            print(f'\tPlotting zoomed in images for {target["dataset"]} (X{target["scale"]})...')

            hr_img_name = self.utility.get_imgs_with_tag_from_dir(dir_path=target['hr_dir'], tag='HR')[self.image_index]
            output_name = f'{IMAGES_DIR}/{target["dataset"]}_X{target["scale"]}_{self.utility.get_img_num(hr_img_name)}_HR.png'
            
            self.___plot_zoomed_img(
                img_path=target['hr_dir'] + '/' + hr_img_name,
                output_path = output_name,
            )

            for model in target['models']:
                up_img_name = self.utility.get_imgs_with_tag_from_dir(dir_path=model['up_dir'], tag=model['tag'])[self.image_index]
                output_name = f'{IMAGES_DIR}/{target["dataset"]}_X{target["scale"]}_{self.utility.get_img_num(up_img_name)}_{model["tag"]}.png'

                self.___plot_zoomed_img(
                    img_path=model['up_dir'] + '/' + up_img_name,
                    output_path = output_name,
                )
        
        print(f'\tSuccessfully plotted zoomed in images. Can be found in {IMAGES_DIR}')

    def plot_epoch_psnr_charts(self):

        print('\n=> Plotting FSRCNN training epoch PSNR charts...')
        plot_targets = {
            'FSRCNN on Track 1 (X4)': [23.99, 25.01, 25.56, 25.75, 25.81, 25.89, 25.85, 25.57, 25.95, 25.94, 25.95, 25.90, 25.90, 25.89, 25.88, 25.95, 25.93, 25.87, 25.95, 25.97], 
            'FSRCNN on Track 2 (X4)': [22.76, 23.54, 23.85, 24.39, 24.54, 24.63, 24.32, 24.38, 24.71, 24.91, 25.00, 25.02, 25.12, 25.14, 25.18, 25.17, 25.13, 25.19, 25.27, 25.30] 
        }

        x_values = [e for e in range(0, 20)]

        fig = plt.gcf()
        fig.set_size_inches(9, 5)

        for label, scores in plot_targets.items():
            plt.plot(x_values, scores, label=label)
            # plt.plot(scores.index(max(scores)), max(scores), label=f'{label} Best PSNR = {max(scores)}')

        plt.xlabel('Epochs')
        plt.ylabel('PSNR Scores')
        plt.xticks(x_values)
        plt.grid()
        plt.legend()
        file_name = f'{PLOTS_DIR}/FSRCNN Training PSNR Scores.png'
        plt.savefig(file_name, bbox_inches='tight', dpi=600)
        plt.close()

        print(f'\tPlotted chart at {file_name}')
            
    def plot_evaluation_charts(self):
        print('\n=> Plotting evaluation charts...')

        for target in TARGETS_EVALUATION:
            print(f'\tPlotting charts for {target["dataset"]} (X{target["scale"]}) evaluation...')

            if not self.utility.file_exists(target['eval_file']):
                print(f'\t\tThe csv file ({target["eval_file"]}) does not exist. Skipping this step.')
                continue
                
            df = pd.read_csv(target['eval_file'])
            
            

            for model in target['models']:
                print(f'\t\t\tPlotting {model["tag"]} charts...')
                print()

                

                



        
