"""Plotter Module"""

import cv2
from matplotlib import cbook
import matplotlib.pyplot as plt
import numpy as np

from helpers.utility import Utility
from helpers.helpers import Helpers
from constants2 import PLOTS_DIR, IMAGES_DIR, TARGETS_EVALUATION

class Plotter: 

    def __init__(self):
        self.utility = Utility()
        self.helpers = Helpers()

        self.utility.check_and_create_dir(PLOTS_DIR)
        self.utility.check_and_create_dir(IMAGES_DIR)

    def ___plot_zoomed_img(self, img_path, output_path, shape=(310, 320)):

        img = cv2.imread(img_path)
        # cv2.imshow('other', img)
        img = self.helpers.crop_to_shape(img, shape)
        # cv2.imshow('cropped', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

        plt.savefig(output_path)


    def plot_zoomed_imgs(self):
        print('\n=> Plotting zoomed images')
        
        for target in TARGETS_EVALUATION[:1]:
            print(f'\tPlotting zoomed in images for {target["dataset"]} (X{target["scale"]})...')

            hr_img_name = self.utility.get_imgs_with_tag_from_dir(dir_path=target['hr_dir'], tag='HR')[0]
            output_name = f'{IMAGES_DIR}/{target["dataset"]}_X{target["scale"]}_{self.utility.get_img_num(hr_img_name)}_HR.png'
            
            self.___plot_zoomed_img(
                img_path=target['hr_dir'] + '/' + hr_img_name,
                output_path = output_name,
            )

            for model in target['models']:
                up_img_name = self.utility.get_imgs_with_tag_from_dir(dir_path=model['up_dir'], tag=model['tag'])[0]
                output_name = f'{IMAGES_DIR}/{target["dataset"]}_X{target["scale"]}_{self.utility.get_img_num(up_img_name)}_{model["tag"]}.png'

                self.___plot_zoomed_img(
                    img_path=model['up_dir'] + '/' + up_img_name,
                    output_path = output_name,
                )
        
        print(f'\tSuccessfully plotted zoomed in images. Can be found in {IMAGES_DIR}')

