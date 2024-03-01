import cv2
import numpy as np
from scipy import stats
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_mabd_mse(video, video2, **kwargs):
    """Calculate MSE of two videos' MABD vector.

    Reference: https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_Learning_to_See_Moving_Objects_in_the_Dark_ICCV_2019_paper.pdf

    Args:
        video (ndarray): Videos with range [0, 255], shape (t, h, w, c), BGR.
        video2 (ndarray): Videos with range [0, 255], shape (t, h, w, c), BGR.

    Returns:
        float: MSE (MABD) result.
    """

    assert video.shape == video2.shape, (f'Video shapes are different: {video.shape}, {video2.shape}.')
    mabd = _MABD(video)
    mabd2 = _MABD(video2)
    return np.mean((mabd - mabd2)**2)


@METRIC_REGISTRY.register()
def calculate_cdc(video, **kwargs):
    """Calculate Color Distribution Consistency index.

    Reference: Temporally Consistent Video Colorization with Deep Feature Propagation and Self-regularization Learning
    https://github.com/bupt-ai-cz/TCVC/blob/main/stage1/utils/cdc.py

    Args:
        video (ndarray): Videos with range [0, 255], shape (t, h, w, c), BGR.

    Returns:
        float: CDC.
    """
    video_list = [video[i] for i in range(len(video))]
    return _cdc(video_list)


def _MABD(video):
    """Calculate mean absolute brightness differences (MABD) for video.

    Args:
        video (ndarray): Videos with range [0, 255], shape (t, h, w, c), BGR.

    Returns:
        ndarray: MABD vector: size of (t-1)
    """
    brightness = 0.114 * video[:, :, :, 0] + 0.587 * video[:, :, :, 1] + 0.299 * video[:, :, :, 2]  # (t, h, w)
    bd = brightness[1:, :, :] - brightness[:-1, :, :]  # (t-1, h, w)
    abd = np.abs(bd)
    mabd = np.mean(abd, axis=(1, 2)) # (t-1)
    return mabd

def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)


def compute_JS_bgr(video_list, dilation=1):
    hist_b_list = []   # [img1_histb, img2_histb, ...]
    hist_g_list = []
    hist_r_list = []

    for img_in in video_list:
        H, W, C = img_in.shape

        hist_b = cv2.calcHist([img_in], [0], None, [256], [0,256]) # B
        hist_g = cv2.calcHist([img_in], [1], None, [256], [0,256]) # G
        hist_r = cv2.calcHist([img_in], [2], None, [256], [0,256]) # R

        hist_b = hist_b / (H * W)
        hist_g = hist_g / (H * W)
        hist_r = hist_r / (H * W)

        hist_b_list.append(hist_b)
        hist_g_list.append(hist_g)
        hist_r_list.append(hist_r)

    JS_b_list = []
    JS_g_list = []
    JS_r_list = []

    for i in range(len(hist_b_list)):
        if i + dilation > len(hist_b_list) - 1:
            break
        hist_b_img1 = hist_b_list[i]
        hist_b_img2 = hist_b_list[i + dilation]
        JS_b = JS_divergence(hist_b_img1, hist_b_img2)
        JS_b_list.append(JS_b)

        hist_g_img1 = hist_g_list[i]
        hist_g_img2 = hist_g_list[i+dilation]
        JS_g = JS_divergence(hist_g_img1, hist_g_img2)
        JS_g_list.append(JS_g)

        hist_r_img1 = hist_r_list[i]
        hist_r_img2 = hist_r_list[i+dilation]
        JS_r = JS_divergence(hist_r_img1, hist_r_img2)
        JS_r_list.append(JS_r)

    return JS_b_list, JS_g_list, JS_r_list


def _cdc(video_list, dilation=[1, 2, 4], weight=[1/3, 1/3, 1/3]):
    # video is a list, each element is an array (cv2.read format)
    mean_b, mean_g, mean_r = 0, 0, 0

    for d, w in zip(dilation, weight):
        JS_b_list_one, JS_g_list_one, JS_r_list_one = compute_JS_bgr(video_list, d)
        mean_b += w * np.mean(JS_b_list_one)
        mean_g += w * np.mean(JS_g_list_one)
        mean_r += w * np.mean(JS_r_list_one)

    cdc = 1/3 * (mean_b + mean_g + mean_r)
    return cdc