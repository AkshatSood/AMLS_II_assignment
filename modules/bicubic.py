"""BicubicInterpolation Module"""

import math

import cv2
import numpy as np


class BicubicInterpolation:
    """Provides functionality to perform bicubic interpolation
    Uses the code provided at 
    https://www.geeksforgeeks.org/python-opencv-bicubic-interpolation-for-resizing-image/
    as reference to create the class.
    """

    def __interpolation_kernel(self, s, a):
        """Interpolation kernal for bicubic interpolation
        """
        if (abs(s) >= 0) & (abs(s) <= 1):
            return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1

        elif (abs(s) > 1) & (abs(s) <= 2):
            return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
        return 0

    def __apply_padding(self, img, H, W, C):
        """Applies padding to the image

        Args:
            img (numpy.ndarray): Raw image
            H (int): Height of the image
            W (int): Width of the image
            C (int): Number of channels in the image

        Returns:
            numpy.ndarray: Image with padding
        """
        zimg = np.zeros((H+4, W+4, C))
        zimg[2:H+2, 2:W+2, :C] = img

        # Pad the first/last two col and row
        zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C]
        zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :]
        zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :]
        zimg[0:2, 2:W+2, :C] = img[0:1, :, :C]

        # Pad the missing eight points
        zimg[0:2, 0:2, :C] = img[0, 0, :C]
        zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C]
        zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C]
        zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C]

        return zimg

    def __bicubic_interpolation(self, img, scale, a):
        """Upscales the provided image with bicubic interpolation

        Args:
            img (numpy.ndarray): Image to be interpolated
            scale (int): Scaling factor
            a (float): Interpolation coefficient (between -0.5 and -0.75) 

        Returns:
            numpy.ndarray: Interpolated image
        """

        # Get image size
        height, width, channels = img.shape
        img = self.__apply_padding(img, height, width, channels)

        # Create new image
        new_height = math.floor(height*scale)
        new_width = math.floor(width*scale)

        interpolated_img = np.zeros((new_height, new_width, 3))

        h = 1/scale

        for c in range(channels):
            for j in range(new_height):
                for i in range(new_width):

                    # Getting the coordinates of the
                    # nearby values
                    x, y = i * h + 2, j * h + 2

                    x1 = 1 + x - math.floor(x)
                    x2 = x - math.floor(x)
                    x3 = math.floor(x) + 1 - x
                    x4 = math.floor(x) + 2 - x

                    y1 = 1 + y - math.floor(y)
                    y2 = y - math.floor(y)
                    y3 = math.floor(y) + 1 - y
                    y4 = math.floor(y) + 2 - y

                    # Considering all nearby 16 values
                    mat_l = np.matrix([[self.__interpolation_kernel(x1, a), self.__interpolation_kernel(
                        x2, a), self.__interpolation_kernel(x3, a), self.__interpolation_kernel(x4, a)]])
                    mat_m = np.matrix([[img[int(y-y1), int(x-x1), c],
                                        img[int(y-y2), int(x-x1), c],
                                        img[int(y+y3), int(x-x1), c],
                                        img[int(y+y4), int(x-x1), c]],
                                       [img[int(y-y1), int(x-x2), c],
                                        img[int(y-y2), int(x-x2), c],
                                        img[int(y+y3), int(x-x2), c],
                                        img[int(y+y4), int(x-x2), c]],
                                       [img[int(y-y1), int(x+x3), c],
                                        img[int(y-y2), int(x+x3), c],
                                        img[int(y+y3), int(x+x3), c],
                                        img[int(y+y4), int(x+x3), c]],
                                       [img[int(y-y1), int(x+x4), c],
                                        img[int(y-y2), int(x+x4), c],
                                        img[int(y+y3), int(x+x4), c],
                                        img[int(y+y4), int(x+x4), c]]])
                    mat_r = np.matrix(
                        [[self.__interpolation_kernel(y1, a)],
                         [self.__interpolation_kernel(y2, a)],
                         [self.__interpolation_kernel(y3, a)],
                         [self.__interpolation_kernel(y4, a)]])

                    interpolated_img[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

        return interpolated_img

    def run(self, input, output, scale):
        """Read the image from the provided path, upscale it and 
        save the image at the provided output path

        Args:
            input (str): Image path
            output (str): Path where the interpolated image needs to be stored
            scale (int): Scaling factor
        """
        img = cv2.imread(input)
        a = -1/2
        interpolated_img = self.__bicubic_interpolation(img, scale, a)
        cv2.imwrite(output, interpolated_img)
