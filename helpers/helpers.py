"""Helpers Module"""

import numpy as np
import torch


class Helpers: 

    def convert_rgb_to_ycbcr(self, img, dim_order='hwc'):
        if dim_order == 'hwc':
            y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
            cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
            cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
        else:
            y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
            cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
            cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])

    def convert_ycbcr_to_rgb(self, img, dim_order='hwc'):
        if dim_order == 'hwc':
            r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
            g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
            b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
        else:
            r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
            g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
            b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])

    def preprocess_for_fsrcnn(self, img, device):
        img = np.array(img).astype(np.float32)
        ycbcr = self.convert_rgb_to_ycbcr(img)
        x = ycbcr[..., 0]
        x /= 255.
        x = torch.from_numpy(x).to(device)
        x = x.unsqueeze(0).unsqueeze(0)
        return x, ycbcr