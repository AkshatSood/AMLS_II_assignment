"""Plotter Module"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
        plt.close()

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

    def __plot_line_chart(self, x, y, xlabel, ylabel, ylim, output):
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ylim)
        plt.xlim((0, len(x)))
        # plt.xticks(x_values)
        plt.grid()
        plt.savefig(output, bbox_inches='tight', dpi=400)
        plt.close()

    def plot_evaluation_charts(self):
        print('\n=> Plotting individual evaluation charts...')

        for target in TARGETS_EVALUATION[:1]:
            print(f'\tPlotting charts for {target["dataset"]} (X{target["scale"]}) evaluation...')

            if not self.utility.file_exists(target['eval_file']):
                print(f'\t\tThe csv file ({target["eval_file"]}) does not exist. Skipping this step.')
                continue
            
            out_dir = f'{PLOTS_DIR}/{target["dataset"]}_X{target["scale"]}'
            self.utility.check_and_create_dir(out_dir)

            df = pd.read_csv(target['eval_file'])
            
            
            for model in target['models'][:1]:
                print(f'\t\tPlotting {model["tag"]} charts...')
                prefix = f'{target["dataset"]}_X{target["scale"]}_{model["tag"]}'

                model_df = df[df.model == model['tag']]
                print(model_df)

                metric_targets = [
                    {
                        'metric': 'RGB_PSNR',
                        'col_name': 'rgb_psnr',
                        'y_label': 'PSNR Scores',
                        'ylim': (0, 50),
                    },
                    {
                        'metric': 'RGB_MSE',
                        'col_name': 'rgb_mse',
                        'y_label': 'Mean Square Error',
                        'ylim': (0, 100),
                    },
                    
                ]
                
                image_numbers = model_df['num'].to_list()

                for mt in metric_targets:
                    print(f'\t\t\tPlotting chart for {mt["metric"]}...')
                    self.__plot_line_chart(
                        x=image_numbers,
                        y=model_df[mt['col_name']].to_list(),
                        xlabel='Image Numbers',
                        ylabel=mt['y_label'],
                        ylim=mt['ylim'],
                        output=f'{out_dir}/{prefix}_{mt["metric"]}.png'
                    )
                


                # plt.plot(image_numbers, psnr_scores)
                # plt.xlabel('Epochs')
                # plt.ylabel('PSNR Scores')
                # plt.ylim((5, 45))
                # # plt.xticks(x_values)
                # plt.grid()
                # plt.savefig('./test.png')
                # plt.close()

                # plt.plot()

    def __plot_correlation_matrices(self, df, cols, labels, title, font_scale, subfolder=None):
        """Plots the correlation matrices from the given data
        Plots both the Pearson's correlation and the Spearman's
        correlation for the provided dataframe. Only the columns
        specified in 'cols' will be used to plot the matrices.
        Args:
            df (pandas.core.frame.DataFrame): Data for correlation matrices
            cols (list): Dataframe column names which need to be plotted
            labels (list): Column name labels
            title (str): Title of the chart
            font_scale (float): The font scale to be used by Seaborn
            subfolder (str, optional): Name of the subfolder where the chart will be stored.
                Defaults to None.
        """
        sns.set(font_scale=font_scale)

        df = df[cols]

        pearson_matrix = df.corr(method='pearson')
        sns.heatmap(
            pearson_matrix,
            fmt='.2f',
            annot=True,
            linewidths=2,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            vmax=1,
            vmin=-1,
        )
        plt.tight_layout()
        plt.title(f'{title} - Pearson\'s Correlation')

        if subfolder is None:
            plt.savefig(f'{PLOTS_DIR}/{title} pearson correlation.png', dpi=600)
        else:
            self.utility.check_and_create_dir(f'{PLOTS_DIR}/{subfolder}')
            plt.savefig(f'{PLOTS_DIR}/{subfolder}/{title} pearson correlation.png', dpi=600)

        plt.close()

        spearman_matrix = df.corr(method='spearman')
        sns.heatmap(
            spearman_matrix,
            fmt='.2f',
            annot=True,
            linewidths=2,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            vmax=1,
            vmin=-1,
        )
        plt.tight_layout()
        plt.title(f'{title} - Spearman\'s Correlation')

        if subfolder is None:
            plt.savefig(f'{PLOTS_DIR}/{title} spearman correlation.png', dpi=600)
        else:
            self.utility.check_and_create_dir(f'{PLOTS_DIR}/{subfolder}')
            plt.savefig(f'{PLOTS_DIR}/{subfolder}/{title} spearman correlation.png', dpi=600)

        plt.close()

    def plot_summary_charts(self):
        print('\n=> Plotting individual evaluation charts...')

        summary_df = None
        
        print('\tCreating summary dataframe....')
        for target in TARGETS_EVALUATION[:6]:
            if not self.utility.file_exists(target['eval_file']):
                print(f'\t\tThe csv file ({target["eval_file"]}) does not exist. Skipping this file.')
                continue
                
            if summary_df is not None:
                df = pd.read_csv(target['eval_file'])
                summary_df = pd.concat([summary_df, df])
            else: 
                summary_df = pd.read_csv(target['eval_file'])

        corr_subfolder = 'correlation'

        # Plot all metrics correlation matrices
        self.__plot_correlation_matrices(
            df=summary_df, 
            cols=['rgb_psnr', 'y_psnr', 'rgb_mse' , 'y_mse', 'ssim', 'uqi', 'hr_entropy', 'up_entropy', 'entropy_perc', 'hr_brisque', 'up_brisque', 'brisque_perc', 'hr_y_brisque', 'up_y_brisque', 'brisque_perc_y'],
            labels=['RGB PSNR', 'Y PSNR', 'RGB MSE', 'Y MSE', 'SSIM', 'UQI', 'HR Entropy', 'UP Entropy', 'Entropy %', 'HR BRISQUE (RGB)', 'UP BRISQUE (RGB)', 'BRISQUE % (RGB)', 'HR BRISQUE (Y)', 'UP BRISQUE (Y)', 'BRISQUE % (Y)'], 
            title='All Metrics',
            font_scale=0.6,
            subfolder=corr_subfolder
        )

        # Plot the FR metrics correlation matrices
        self.__plot_correlation_matrices(
            df=summary_df, 
            cols=['rgb_psnr', 'y_psnr', 'rgb_mse' , 'y_mse', 'ssim', 'uqi'],
            labels=['PSNR (RGB)', 'PSNR (Y)', 'MSE (RGB)', 'MSE (Y)', 'SSIM', 'UQI'], 
            title='Full-Reference Metrics',
            font_scale=0.9,
            subfolder=corr_subfolder
        )

        # Plot the NR metrics correlation matrices
        self.__plot_correlation_matrices(
            df=summary_df, 
            cols=['hr_entropy', 'up_entropy', 'entropy_perc', 'hr_brisque', 'up_brisque', 'brisque_perc', 'hr_y_brisque', 'up_y_brisque', 'brisque_perc_y'],
            labels=['HR Entropy', 'UP Entropy', 'Entropy %', 'HR BRISQUE (RGB)', 'UP BRISQUE (RGB)', 'BRISQUE % (RGB)', 'HR BRISQUE (Y)', 'UP BRISQUE (Y)', 'BRISQUE % (Y)'], 
            title='No-Reference Metrics',
            font_scale=0.8,
            subfolder=corr_subfolder
        )

        
            
        



                





                

                



        
