import cv2
import math
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.matlab_functions import imresize
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_uciqe(img, crop_border):
    """Calculate UCIQE (Undewater Color Image Quality Evaluation) metric.

    ``Paper: An underwater color image quality evaluation metric``

    https://github.com/JOU-UIP/UCIQE/blob/main/UCIQE.py

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.

    Returns:
        float: UCIQE result.
    """

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    coe_metric = [0.4680, 0.2745, 0.2576]  # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    uciqe_result = coe_metric[0]*var_chr + coe_metric[1]*con_lum + coe_metric[2]*aver_sat # get final quality value

    return uciqe_result